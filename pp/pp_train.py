import math, os, argparse, torch, torch.nn as nn, torch.nn.functional as F
import deepspeed
import torch
from deepspeed.pipe import PipelineModule, LayerSpec, TiedLayerSpec
from torch.utils.data import Dataset, DataLoader


class Qwen3InputPipe(nn.Module):
    def __init__(self, tie_embed: nn.Embedding):
        super().__init__()
        self.embed_tokens = tie_embed
    def forward(self, batch:dict[str, torch.Tensor])->tuple:
        input_ids        = batch["input_ids"]
        attention_mask   = batch.get("attention_mask", None)
        position_ids     = batch.get("position_ids", None)
        hidden_states = self.embed_tokens(input_ids)
        return (hidden_states, attention_mask, position_ids)

class Qwen3OutputPipe(nn.Module):
    def __init__(self, tie_lm_head: nn.Linear, rms_norm):
        super().__init__()
        self.norm = rms_norm
        self.lm_head = tie_lm_head
    def forward(self, x):
        hidden_states, *_ = x
        hs = self.norm(hidden_states)
        logits = self.lm_head(hs)   # [B, S, vocab]
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
    """takes in a tuple, unpack it to call Qwen3DecoderLayer, then pass everything down"""
    def __init__(self, decoder_block, layer_idx: int):
        super().__init__()
        self.block = decoder_block
        self.layer_idx = layer_idx

    def forward(self, x):
        # tuple structure convention: hidden_states, attention_mask are required, others are optional
        # (hidden_states, attention_mask, position_ids, past_key_values, use_cache, cache_position, position_embeddings)
        hs, am, pid, pkv, use_cache, cp, pos_emb, kw = _normalize_tuple(x)
        hs = self.block(
            hidden_states=hs, attention_mask=am, position_ids=pid,
            past_key_values=pkv, use_cache=bool(use_cache),
            cache_position=cp, position_embeddings=pos_emb, **kw
        )
        return (hs, am, pid, pkv, use_cache, cp, pos_emb, kw)

def build_pipeline(args):
    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
    print("passin dtype", dtype)
    tok = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    qwen3_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        dtype=dtype,
    )
    decoder_layers = qwen3_model.model.layers
    n_layer = len(decoder_layers)
    d_model = qwen3_model.model.embed_tokens.embedding_dim
    pad_id  = tok.pad_token or tok.eos_token

    emb_tied = emb_tied = TiedLayerSpec(
        "tok_embed",              # tied_id，input output should be the same
        nn.Embedding, vocab_size, d_model, padding_idx=pad_id)
    lm_tied  = TiedLayerSpec("tok_embed", nn.Linear, hidden, vocab_size, bias=False)

    layers = []
    layers.append(TiedLayerSpec("tok_embed", qwen3_model.model.embed_tokens, qwen3_model.model.vocab_size, d_model, args.seq_len))
    for _ in range(n_layer // 2):
        layers.append(LayerSpec(Block, d_model, n_head, 4))
    for _ in range(n_layer // 2):
        layers.append(LayerSpec(Block, d_model, n_head, 4))
    head = LayerSpec(LMHead, d_model, vocab)
    layers.append(head)

    




    # loss_fn applied in the end
    def loss_fn(outputs, labels):
        # outputs: [B,T,V], labels: [B,T]
        logits = outputs
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=0)
        return loss

    pipe = PipelineModule(
        layers=layers,
        num_stages=2,
        loss_fn=loss_fn,
        partition_method="parameters",          # 均分参数量到两个 stage
        activation_checkpoint_interval=0        # 先关闭重算（可改为 >0 开启）
    )

    # 绑定 LMHead 的权重共享（与 tok embedding）
    # 注意：DeepSpeed 的 TiedLayerSpec 会在构建时绑定相同名字的参数
    return pipe


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

