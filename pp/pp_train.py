import math, os, argparse, json, torch, torch.nn as nn, torch.nn.functional as F
import deepspeed
import torch
from deepspeed.pipe import PipelineModule, LayerSpec, TiedLayerSpec
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm, Qwen3DecoderLayer
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
        # select per-layer mask for computation; keep original mapping for downstream layers
        am_for_layer = am.get(self.block.attention_type, None) if isinstance(am, dict) else am
        hs = self.block(
            hidden_states=hs, attention_mask=am_for_layer, position_ids=pid,
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
        model_name_or_path, dtype=dtype, device_map="cpu", low_cpu_mem_usage=True
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
        loss_fn=lambda logits, labels: pp_safe_ce_loss(logits, labels, ignore_index=-100, label_smoothing=0.0),
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
        tail = next((m for m in pipe.modules() if isinstance(m, Qwen3OutputPipe)), None)
        if tail is not None:
            _copy(tail.norm.weight, "model.norm.weight")
            ok = _copy(tail.head.weight, "lm_head.weight")
            if not ok:  # 兜底：某些权重 tying 的模型
                _copy(tail.head.weight, "model.embed_tokens.weight")


class JsonlCausalDataset(Dataset):
    """最简单的数据集：读取 JSONL，每行含 {input: "...", target: "..."}，拼成 input_ids / labels"""
    def __init__(self, path, tokenizer, input_key="input", target_key="target", seq_len=1024):
        self.items = [json.loads(l) for l in open(path, "r", encoding="utf-8")]
        self.tok = tokenizer
        self.ikey, self.tkey = input_key, target_key
        self.S = seq_len
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        ex = self.items[idx]
        text = str(ex[self.ikey]) + str(ex[self.tkey])
        ids  = self.tok(text, truncation=True, max_length=self.S, padding="max_length", return_tensors="pt").input_ids[0]
        # 右移标签：预测下一个 token；第一个位置无标签置 -100
        labels = ids.clone()
        labels[:-1] = ids[1:]
        labels[-1]  = -100
        am = (ids != self.pad_id).long()
        return {"input_ids": ids, "attention_mask": am}, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deepspeed_config", type=str, default="ds_pp_2gpus.json")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--train_json", type=str, required=True)
    parser.add_argument("--input_key", type=str, default="input")
    parser.add_argument("--target_key", type=str, default="target")
    parser.add_argument("--seq_len", type=int, default=1024)
    # Added to be compatible with DeepSpeed launcher
    parser.add_argument("--local_rank", type=int, default=-1)
    prec = parser.add_mutually_exclusive_group()
    prec.add_argument("--bf16", action="store_true")
    prec.add_argument("--fp16", action="store_true")
    parser.add_argument("--micro_batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--log_every", type=int, default=20)
    args = parser.parse_args()

    torch.manual_seed(42)
    local_rank = args.local_rank
    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)

    # 1) 只加载一次 HF 模型 + 构建 2-stage Pipeline
    pipe, ref = build_pipeline_and_ref(args.model_name_or_path, dtype)

    # 2) 初始化 DeepSpeed（注意：这里传的 config 是 JSON 路径或 dict，按你环境来）
    engine, _, _, _ = deepspeed.initialize(
        model=pipe,
        model_parameters=[p for p in pipe.parameters() if p.requires_grad],
        config=args.deepspeed_config,
    )

    # 3) **就是这里**：把这一次加载的权重拷到当前 rank 的子网
    copy_qwen3_weights_to_pipe(engine, ref)
    del ref  # 需要的话释放内存

    # 4) 数据
    #    - Pipeline 的第一个 stage 期望拿到 dict（input_ids/attention_mask 等），
    #    - 最后一段会拿到 labels（engine.train_batch(data=(batch_dict, labels))）
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    ds  = JsonlCausalDataset(args.train_json, tok, args.input_key, args.target_key, args.seq_len)
    # ds = JsonlSFTDataset(args.train_json, input_key=args.input_key, target_key=args.target_key)
    dl  = DataLoader(ds, batch_size=engine.train_micro_batch_size_per_gpu(), shuffle=True, drop_last=True)

    # 5) 训练循环
    step = 0
    for _ in range(args.epochs):
        for batch_dict, labels in dl:
            loss = engine.train_batch(data=(batch_dict, labels))
            step += 1
            if engine.is_gradient_accumulation_boundary() and engine.global_rank == 0 and (step % args.log_every == 0):
                print(f"step {step}  loss {loss.item():.4f}")


if __name__ == "__main__":
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
    main()

