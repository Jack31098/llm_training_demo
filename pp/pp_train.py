import math, os, argparse, torch, torch.nn as nn, torch.nn.functional as F
import deepspeed
import torch
from deepspeed.pipe import PipelineModule, LayerSpec, TiedLayerSpec
from torch.utils.data import Dataset, DataLoader
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
from input_pipe import Qwen3InputPipe


class Qwen3OutputPipe(nn.Module):
    def __init__(self, d_model: int, vocab: int, eps: float = 1e-6):
        super().__init__()
        self.norm = Qwen3RMSNorm(d_model, eps=eps)
        self.head = nn.Linear(d_model, vocab, bias=False)

    def forward(self, x):
        hs = x["hidden_states"] if isinstance(x, dict) else (x[0] if isinstance(x, (tuple, list)) else x)
        return self.head(self.norm(hs))  # [B,S,V]

def _normalize_tuple(x):
    # standalize to a tuple of length 8，the last one is a kwargs dict
    if isinstance(x, torch.Tensor):
        x = (x, None, None, None, False, None, None, {})
    else:
        x = tuple(x)
        if len(x) < 8:
            x = (x + (None,)*(8-len(x)))  # padding to length 8
        if x[7] is None:    # if a kwarg passin, it won't be None
            x = x[:7] + ({},)
    return x

class Qwen3DecoderLayerPipe(nn.Module):
    """Wrap Qwen3DecoderLayer to accept/return a tuple for pipeline."""
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.block = Qwen3DecoderLayer(config, layer_idx)
        self.layer_idx = layer_idx

    def forward(self, x):
        # tuple structure convention: hidden_states, attention_mask are required, others are optional 
        # (hidden_states, attention_mask, position_ids, past_key_values, use_cache, cache_position, position_embeddings, kw)
        hs, am, pid, pkv, use_cache, cp, pos_emb, kw = _normalize_tuple(x)
        hs = self.block(
            hidden_states=hs, attention_mask=am, position_ids=pid,
            past_key_values=pkv, use_cache=bool(use_cache),
            cache_position=cp, position_embeddings=pos_emb, **kw
        )
        return (hs, am, pid, pkv, use_cache, cp, pos_emb, kw)

def pp_safe_ce_loss(
    logits: torch.Tensor,          # [B, S, V] —— last stage output
    labels: torch.Tensor,          # [B, S]     —— Pipeline will send it to the last stage
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """
    numerically stable Causal LM cross entropy, used for DeepSpeed Pipeline's loss_fn.
    - automatically shift right (if labels and logits sequence length are the same)
    - only compute on valid positions (labels != ignore_index)
    - compute CE in FP32; clean logits to avoid NaN/Inf propagation
    """
    # ---- 1) clean &提升精度（只在 loss 内部做，计算图仍连通）----
    if not torch.is_floating_point(logits):
        logits = logits.float()
    logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)

    # ---- 2) shift right (most common HF behavior)----
    if labels.dim() == 2 and logits.size(1) == labels.size(1):
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
    else:
        # if your data pipeline has aligned and set invalid positions to -100, you can shift right
        shift_logits, shift_labels = logits, labels

    # ---- 3) flatten + valid position mask ----
    V = shift_logits.size(-1)
    flat_logits = shift_logits.view(-1, V)
    flat_labels = shift_labels.view(-1)
    valid = flat_labels != ignore_index

    if not torch.any(valid):
        # all positions are ignored, return 0 (scalar), keep the gradient graph complete
        return flat_logits.new_zeros((), dtype=torch.float32, requires_grad=True)

    # ---- 4) compute CE on valid positions (support label smoothing)----
    if label_smoothing > 0.0:
        loss = F.cross_entropy(
            flat_logits[valid],
            flat_labels[valid],
            reduction="mean",
            label_smoothing=label_smoothing,
        )
    else:
        loss = F.cross_entropy(
            flat_logits[valid],
            flat_labels[valid],
            reduction="mean",
        )
    return loss


def dbg_full_model_safe_ce_loss(model, input_ids, attention_mask, labels, log_every=None, step=None, is_master=True):
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits
    if is_master and log_every is not None and (step is None or step % log_every == 0 or step <= 5):
        m = logits.mean().item(); a = logits.amin().item(); b = logits.amax().item()
        print({"debug_logits": {"mean": round(m,2), "min": round(a,2), "max": round(b,2)}})

    # reuse the same implementation
    return pp_safe_ce_loss(logits, labels, ignore_index=-100, label_smoothing=0.0)


def build_pipeline_and_ref(model_name_or_path: str, dtype: torch.dtype):
    # —— only load once ——（get config and state_dict）
    ref = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, torch_dtype=dtype, device_map="cpu", low_cpu_mem_usage=True
    )
    tok = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    cfg      = ref.config
    n_layer  = cfg.num_hidden_layers
    d_model  = cfg.hidden_size
    vocabsz  = cfg.vocab_size
    has_swa  = ("sliding_attention" in getattr(cfg, "layer_types", []))

    layers = []
    # stage0 entry point：embed/rope/mask等全部在 InputPipe 里做（后面拷权）
    layers.append(LayerSpec(Qwen3InputPipe, cfg, vocabsz, d_model, pad_id, has_swa, True))
    # intermediate decoder layers
    for lid in range(n_layer):
        layers.append(LayerSpec(Qwen3DecoderLayerPipe, cfg, lid))
    # stage1 tail: Norm → Linear（lm_head），后面用“值拷贝”对齐
    layers.append(LayerSpec(Qwen3OutputPipe, d_model, vocabsz))

    pipe = PipelineModule(
        layers=layers,
        num_stages=2,                      # hardcoded two stages
        partition_method="uniform",        # simple: split by layer number
        loss_fn=loss_fn=lambda logits, labels: pp_safe_ce_loss(logits, labels, ignore_index=-100, label_smoothing=0.0),
        activation_checkpoint_interval=0
    )
    return pipe, ref

@torch.no_grad()
def copy_qwen3_weights_to_pipe(engine, ref):
    """copy the weights of Qwen3ForCausalLM(ref) to the current rank's Pipeline subnet"""
    ref_sd = ref.state_dict()
    pipe   = engine.module

    def _copy(dst, key):
        src = ref_sd.get(key, None)
        if src is None or tuple(src.shape) != tuple(dst.shape):
            return False
        dst.copy_(src); return True

    # stage0：InputPipe 的 embedding / rotary
    if engine.is_first_stage():
        ip = next((m for m in pipe.modules() if isinstance(m, Qwen3InputPipe)), None)
        if ip is not None:
            _copy(ip.embed_tokens.weight, "model.embed_tokens.weight")
            # rotary (mostly buffer)
            ip.rotary_emb.load_state_dict(ref.model.rotary_emb.state_dict(), strict=False)

    # all decoder layers: copy by layer name
    for mod in pipe.modules():
        if isinstance(mod, Qwen3DecoderLayerPipe):
            g   = int(getattr(mod, "layer_idx"))
            pfx = f"model.layers.{g}."
            blk = mod.block
            _copy(blk.self_attn.q_proj.weight,        pfx+"self_attn.q_proj.weight")
            _copy(blk.self_attn.k_proj.weight,        pfx+"self_attn.k_proj.weight")
            _copy(blk.self_attn.v_proj.weight,        pfx+"self_attn.v_proj.weight")
            _copy(blk.self_attn.o_proj.weight,        pfx+"self_attn.o_proj.weight")
            _copy(blk.self_attn.q_norm.weight,        pfx+"self_attn.q_norm.weight")
            _copy(blk.self_attn.k_norm.weight,        pfx+"self_attn.k_norm.weight")
            _copy(blk.input_layernorm.weight,         pfx+"input_layernorm.weight")
            _copy(blk.post_attention_layernorm.weight,pfx+"post_attention_layernorm.weight")
            _copy(blk.mlp.gate_proj.weight,           pfx+"mlp.gate_proj.weight")
            _copy(blk.mlp.up_proj.weight,             pfx+"mlp.up_proj.weight")
            _copy(blk.mlp.down_proj.weight,           pfx+"mlp.down_proj.weight")

    # stage1：Norm → lm_head（用“值拷贝”，不做指针 tying）
    if engine.is_last_stage():
        # NormPipe
        np = next((m for m in pipe.modules() if isinstance(m, NormPipe)), None)
        if np is not None:
            _copy(np.norm.weight, "model.norm.weight")
        # last layer linear layer as lm_head
        head = next((m for m in pipe.modules()
                     if isinstance(m, nn.Linear)
                     and m.in_features == ref.config.hidden_size
                     and m.out_features == ref.config.vocab_size), None)
        if head is not None:
            # directly copy lm_head；if the model binds head to embed, the values are already the same
            ok = _copy(head.weight, "lm_head.weight")
            if not ok:  # fallback: use embed weights
                _copy(head.weight, "model.embed_tokens.weight")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deepspeed_config", type=str, default="ds_pp_2gpus.json")
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--train_json", type=str, required=True)
    p.add_argument("--input_key", type=str, default="input")
    p.add_argument("--target_key", type=str, default="target")
    # training
    p.add_argument("--seq_len", type=int, default=1024)
    prec = p.add_mutually_exclusive_group()
    prec.add_argument("--bf16", action="store_true", help="Enable bfloat16 training")
    prec.add_argument("--fp16", action="store_true", help="Enable float16 training")
    p.add_argument("--use_activation_checkpointing", action="store_true")
    p.add_argument("--pack_sequences", action="store_false")
    p.add_argument("--micro_batch_size", type=int, default=1)
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--wd", type=float, default=0.1)
    p.add_argument("--max_grad_norm", type=float, default=0.1)
    p.add_argument("--max_steps", type=int, default=100)
    # debugging/toggles
    p.add_argument("--debug_autograd", action="store_true", help="Enable autograd anomaly detection (slow)")
    p.add_argument("--sanitize_grads", action="store_true", help="Replace NaN/Inf grads with 0 and clamp gradients")
    p.add_argument("--attn_impl", type=str, default="sdpa", choices=["eager", "sdpa"], help="Attention implementation")
    p.add_argument("--debug_nan", action="store_true", help="Enable per-layer forward NaN/Inf detection on rank0")
    # logging / io
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--debug_per_sample_loss", action="store_true")
    p.add_argument("--debug_every", type=int, default=100)
    p.add_argument("--save_ckpt_every", type=int, default=0)
    p.add_argument("--output_dir", type=str, default="runs/demo")
    p.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    torch.manual_seed(42)

    model = build_pipeline(args.vocab, args.d_model, args.n_head, args.n_layer, args.seq_len)

    engine, _, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=[p for p in model.parameters() if p.requires_grad],
        config=args.deepspeed_config
    )

    # simple data iteration (DeepSpeed Pipeline directly feeds (input, labels))
    dataset = ToyDataset(args.samples, args.seq_len, args.vocab)
    loader = DataLoader(dataset, batch_size=engine.train_micro_batch_size_per_gpu(), shuffle=True, drop_last=True)

    for epoch in range(args.epochs):
        for it, (x, y) in enumerate(loader):
            # DeepSpeed Pipeline 的输入是 tuple；引擎负责把 labels 送到最后一段
            loss = engine.train_batch(data=(x, y))
            if engine.is_gradient_accumulation_boundary():
                if engine.global_rank == 0:
                    print(f"epoch {epoch} iter {it} loss {loss.item():.4f}")

if __name__ == "__main__":
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
    main()

