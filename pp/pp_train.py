import math, os, argparse, torch, torch.nn as nn, torch.nn.functional as F
import deepspeed
import torch
from deepspeed.pipe import PipelineModule, LayerSpec, TiedLayerSpec
from torch.utils.data import Dataset, DataLoader
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
from input_pipe import Qwen3InputPipe


class Qwen3OutputPipe(nn.Module):
    def __init__(self, d_model: int, vocab: int, tie_weight: Optional[nn.Parameter] = None):
        super().__init__()
        self.norm = Qwen3RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab, bias=False)
        if tie_weight is not None:
            # 单进程/同 stage 才能共享同一 Parameter
            self.lm_head.weight = tie_weight  # 不要 .clone() / .detach()

    def forward(self, x):
        # DeepSpeed 可能把 (hidden_states, labels) 作为 tuple 传到最后一段
        hidden_states = x[0] if isinstance(x, (tuple, list)) else x
        logits = self.lm_head(self.norm(hidden_states))   # [B, S, vocab]
        return logits

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

def build_pipeline(args):
    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)

    tok = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    # load the model on cpu first
    # this is not the general way to load model
    # when the model is very large, any rank should load its own params
    qwen3 = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        device_map="cpu",               #  config + state_dict for later loading into gpu
        low_cpu_mem_usage=True,
    )
    cfg      = qwen3.config
    n_layer  = cfg.num_hidden_layers
    d_model  = cfg.hidden_size
    vocabsz  = cfg.vocab_size

    # TiedLayerSpec：same name= "tok_embed" it means different stages share the same weight
    embed_tied = TiedLayerSpec("tok_embed", torch.nn.Embedding, vocabsz, d_model, pad_id)
    head_tied  = TiedLayerSpec("tok_embed", torch.nn.Linear,    d_model, vocabsz, bias=False)

    layers = []
    layers.append(embed_tied)                                   # ① 嵌入在最前
    for lid in range(n_layer):
        layers.append(LayerSpec(Qwen3DecoderLayerPipe, cfg, lid))
    layers.append(LayerSpec(Qwen3OutputPipe, d_model, vocabsz))      # ② 尾部 Norm→Head（head 会被拷权）

    # 简单 CE（Pipeline 会把 labels 送到最后一段的 loss_fn）
    def loss_fn(logits, labels):
        return F.cross_entropy(logits.view(-1, vocabsz), labels.view(-1), ignore_index=pad_id)

    pipe = PipelineModule(
        layers=layers,
        num_stages=2,
        loss_fn=loss_fn,
        partition_method="parameters",  # 或 "uniform"；DeepSpeed 目前没有稳定的“逐层手工分配”API
        activation_checkpoint_interval=0
    )

    return pipe, qwen3


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

    # 简单数据迭代（DeepSpeed Pipeline 直接喂 (input, labels)）
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

