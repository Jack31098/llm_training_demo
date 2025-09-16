import sys, math, os, argparse, json, time, torch, torch.nn as nn, torch.nn.functional as F
import deepspeed
from torch.amp import autocast
import torch
import torch.distributed as dist
import torch.utils.checkpoint as ckpt
from deepspeed.pipe import PipelineModule, LayerSpec, TiedLayerSpec
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm, Qwen3DecoderLayer, create_causal_mask, create_sliding_window_causal_mask
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
from input_pipe import Qwen3InputPipe, _assert_pipe_tuple
from torch.profiler import profile, ProfilerActivity, schedule


iter = 0

class Qwen3OutputPipe(nn.Module):
    def __init__(self, d_model: int, vocab: int, eps: float = 1e-6):
        super().__init__()
        self.norm = Qwen3RMSNorm(d_model, eps=eps)
        self.head = nn.Linear(d_model, vocab, bias=False)
        # expose top-level weight for TiedLayerSpec
        self.weight = self.head.weight

    def forward(self, x):
        if isinstance(x, (tuple, list)):
            hs, _pad, _pid, _cp, _r1, _r2 = x
        else:
            hs = x
        return self.head(self.norm(hs))  # [B,S,V]


class Qwen3DecoderLayerPipe(nn.Module):
    """Wrap Qwen3DecoderLayer to accept/return a tuple for pipeline."""
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.block = Qwen3DecoderLayer(config, layer_idx)
        self.config = config
        self.layer_idx = layer_idx
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self.use_ckpt = True

    def forward(self, x):
        hs, pad_mask, pid, cp, r1, r2 = x
        # 确保进入 checkpoint 的第一个实参有梯度（hs 是 leaf，允许就地置位）
        if hs.is_leaf and hs.is_floating_point() and not hs.requires_grad:
            hs.requires_grad_()
        if self.use_ckpt:
            return ckpt.checkpoint(self._forward, x, use_reentrant=False)
        else:
            return self._forward(x)

    def _forward(self, x):
        # with autocast("cuda", dtype=torch.bfloat16, enabled=True):
        # 6-slot tensors-only convention (no floats besides hs):
        # (hidden_states, padding_mask, position_ids, cache_position, r1, r2)
        # _assert_pipe_tuple(x, f"layer{self.layer_idx}.in")
        hs, pad_mask, pid, cp, _r1, _r2 = x

        # Recompute per-layer attention mask and RoPE to avoid sending complex objects
        am_for_layer = None
        if pad_mask is not None and pid is not None and cp is not None:
            mask_kwargs = dict(
                config=self.config,
                input_embeds=hs,
                attention_mask=pad_mask,
                cache_position=cp,
                past_key_values=None,
                position_ids=pid,
            )
            if getattr(self.block, 'attention_type', 'full_attention') == 'sliding_attention' and create_sliding_window_causal_mask is not None:
                am_for_layer = create_sliding_window_causal_mask(**mask_kwargs)
            else:
                am_for_layer = create_causal_mask(**mask_kwargs)

        # Recompute RoPE per layer locally to avoid sending float ancillaries
        cos, sin = self.rotary_emb(hs, pid)
        hs = self.block(
            hidden_states=hs,
            attention_mask=am_for_layer,
            position_ids=pid,
            past_key_values=None,
            use_cache=False,
            cache_position=cp,
            position_embeddings=(cos, sin) if cos is not None else None,
        )
        ret = (hs, pad_mask, pid, cp, _r1, _r2)
        # _assert_pipe_tuple(ret, f"layer{self.layer_idx}.out")
        return ret

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
    # import pdb; pdb.set_trace()
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


def make_debug_loss_fn(log_every: int = 1, print_logits_stats: bool = True):
    """Create a loss_fn compatible with PipelineModule that prints microbatch loss.
    
    Prints every `log_every` invocations on rank 0. Optionally prints logits stats.
    """
    def _loss_fn(logits: torch.Tensor, labels: torch.Tensor):
        loss = pp_safe_ce_loss(logits, labels, ignore_index=-100, label_smoothing=0.0)
        global iter
        iter += 1
        try:
            should_print = (
                (log_every is not None and log_every > 0 and (iter % log_every == 0))
                or (iter <= 5)  # always print first few for quick sanity
            )
            if should_print:
                if print_logits_stats:
                    lt = logits.detach()
                    print({
                        "micro_batch_iter": iter,
                        "train/mb_loss": float(loss.detach().item()),
                        "logits_mean": float(lt.mean().item()),
                        "logits_min": float(lt.amin().item()),
                        "logits_max": float(lt.amax().item()),
                    }, flush=True)
                else:
                    print({"micro_batch_iter": iter, "train/mb_loss": float(loss.detach().item())}, flush=True)
        except Exception:
            pass
        return loss

    return _loss_fn

def build_tokenizer(model_name_or_path: str):
    """Build a single tokenizer instance and ensure pad_token_id exists.

    Returns
    -------
    tokenizer : PreTrainedTokenizerBase
    pad_id    : int
    """
    tok = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    # Ensure pad token exists for causal LMs (many don't set it by default)
    if tok.pad_token_id is None:
        if tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        else:
            # Fallback: add a PAD token if tokenizer truly lacks eos (rare)
            tok.add_special_tokens({"pad_token": "<|pad|>"})
    pad_id = tok.pad_token_id
    return tok, pad_id


def build_pipeline_and_ref(model_name_or_path: str, dtype: torch.dtype, pad_id: int):
    # —— only load once ——（get config and state_dict）
    ref = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, dtype=dtype, device_map="cpu", low_cpu_mem_usage=True
    )
    tie_word_embeddings = ref.config.tie_word_embeddings
    # if torch.distributed.get_rank() == 0:
    #     import pdb; pdb.set_trace()
    # else:
    #     time.sleep(1000)
    cfg      = ref.config
    n_layer  = cfg.num_hidden_layers
    d_model  = cfg.hidden_size
    vocabsz  = cfg.vocab_size
    has_swa  = ("sliding_attention" in getattr(cfg, "layer_types", []))

    layers = []
    # stage0 entry point: tie token embed with output head via TiedLayerSpec
    layers.append(TiedLayerSpec('tok_embed', Qwen3InputPipe, cfg, vocabsz, d_model, pad_id, has_swa, True))
    # layers.append(LayerSpec(Qwen3InputPipe, cfg, vocabsz, d_model, pad_id, has_swa, True))
    
    # intermediate decoder layers
    for lid in range(n_layer):
        layers.append(LayerSpec(Qwen3DecoderLayerPipe, cfg, lid))
    # stage1 tail: Norm → Linear（lm_head），tied with input embedding
    layers.append(TiedLayerSpec('tok_embed', Qwen3OutputPipe, d_model, vocabsz))
    # layers.append(LayerSpec(Qwen3OutputPipe, d_model, vocabsz))

    pipe = PipelineModule(
        layers=layers,
        num_stages=2,                      # hardcoded two stages
        partition_method="uniform",        # simple: split by layer number
        loss_fn=lambda logits, labels: pp_safe_ce_loss(logits, labels, ignore_index=-100, label_smoothing=0.0),
        activation_checkpoint_interval=0
    )
    return pipe, ref

@torch.no_grad()
def load_from_ref_into_pipe_before_ds_init(pipe, ref):
    ref_sd = ref.state_dict()

    def _copy(dst, key):
        src = ref_sd.get(key, None)
        if src is None or tuple(src.shape) != tuple(dst.shape):
            return False
        dst.copy_(src.to(device=dst.device, dtype=dst.dtype))
        return True

    # InputPipe：embedding / rotary
    ip = next((m for m in pipe.modules() if isinstance(m, Qwen3InputPipe)), None)
    if ip is not None:
        _copy(ip.embed_tokens.weight, "model.embed_tokens.weight")
        ip.rotary_emb.load_state_dict(ref.model.rotary_emb.state_dict(), strict=False)

    # Decoder 全层
    for mod in pipe.modules():
        if isinstance(mod, Qwen3DecoderLayerPipe):
            g = int(getattr(mod, "layer_idx"))
            p = f"model.layers.{g}."
            blk = mod.block
            _copy(blk.self_attn.q_proj.weight,         p+"self_attn.q_proj.weight")
            _copy(blk.self_attn.k_proj.weight,         p+"self_attn.k_proj.weight")
            _copy(blk.self_attn.v_proj.weight,         p+"self_attn.v_proj.weight")
            _copy(blk.self_attn.o_proj.weight,         p+"self_attn.o_proj.weight")
            _copy(blk.self_attn.q_norm.weight,         p+"self_attn.q_norm.weight")
            _copy(blk.self_attn.k_norm.weight,         p+"self_attn.k_norm.weight")
            _copy(blk.input_layernorm.weight,          p+"input_layernorm.weight")
            _copy(blk.post_attention_layernorm.weight, p+"post_attention_layernorm.weight")
            _copy(blk.mlp.gate_proj.weight,            p+"mlp.gate_proj.weight")
            _copy(blk.mlp.up_proj.weight,              p+"mlp.up_proj.weight")
            _copy(blk.mlp.down_proj.weight,            p+"mlp.down_proj.weight")

    # Output：norm + lm_head（这里只对齐一次，不再在 init 之后 copy）
    tail = next((m for m in pipe.modules() if isinstance(m, Qwen3OutputPipe)), None)
    if tail is not None:
        _copy(tail.norm.weight, "model.norm.weight")
        ok = _copy(tail.head.weight, "lm_head.weight")
        if not ok:
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
        out = self.tok(text, truncation=True, max_length=self.S, padding="max_length", return_tensors="pt")
        ids  = out.input_ids[0]
        am   = (ids != self.pad_id).long()
        # 标签与输入对齐；仅将 PAD 位置置为 -100，具体 shift-right 在 loss 内部完成
        labels = ids.clone()
        labels[ids == self.pad_id] = -100
        # Return DS pipeline-compatible structure: ((input_ids, attention_mask), labels)
        return (ids, am), labels


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
    parser.add_argument("--max_examples", type=int, default=1024,
                        help="stop after this many examples")
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--profile_steps", type=int, default=64,
                        help="when >0, sample to calculate T_comp/T_comm")
    parser.add_argument("--profile_trace_dir", type=str, default="",
                        help="optional: write Chrome trace to directory for visualization")
    args = parser.parse_args()

    torch.manual_seed(42)
    # Ensure distributed is initialized before constructing PipelineModule
    local_rank = args.local_rank
    # if not dist.is_initialized():
    #     print(f"Initializing distributed process group with backend nccl on rank {local_rank}")
    #     dist.init_process_group(backend="nccl")
    # Initialize DeepSpeed comm backend (required by PipelineModule)
    deepspeed.init_distributed(dist_backend="nccl")
    print(f"Distributed process group initialized: {dist.is_initialized()} on rank {local_rank}")
    if torch.cuda.is_available() and local_rank >= 0:
        torch.cuda.set_device(local_rank)
    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)

    # 1) 构建 tokenizer（一次），并据此构建 2-stage Pipeline
    tokenizer, pad_id = build_tokenizer(args.model_name_or_path)
    pipe, ref = build_pipeline_and_ref(args.model_name_or_path, dtype, pad_id)
    # Replace loss_fn with a debug-printing variant on demand
    pipe.loss_fn = make_debug_loss_fn(log_every=args.log_every, print_logits_stats=True)

    # 2) 初始化 DeepSpeed，并交给 DeepSpeed 内部去构建分布式 DataLoader（使用同一个 tokenizer）
    ds  = JsonlCausalDataset(args.train_json, tokenizer, args.input_key, args.target_key, args.seq_len)
    load_from_ref_into_pipe_before_ds_init(pipe, ref)
    del ref
    # base_opt = torch.optim.AdamW(
    #     [p for p in pipe.parameters() if p.requires_grad],
    #     lr=5e-6, betas=(0.9, 0.95), eps=1e-6, weight_decay=0.0
    # )

    engine, _, _, _ = deepspeed.initialize(
        model=pipe,
        # model_parameters=[p for p in pipe.parameters() if p.requires_grad],
        model_parameters=None,
        config=args.deepspeed_config,
        training_data=ds,
    )

    # 3) **就是这里**：把这一次加载的权重拷到当前 rank 的子网
    # copy_qwen3_weights_to_pipe(engine, ref)

    # 4) 数据由 DeepSpeed 内部的 RepeatingLoader/DistributedSampler 驱动

    def _is_comm_key(name: str) -> bool:
        n = name.lower()
        return (
            "nccl" in n
            or "all_reduce" in n or "allreduce" in n
            or "reduce_scatter" in n
            or "all_gather" in n or "_all_gather_base" in n
            or "broadcast" in n
            or ("send" in n and "nccl" in n) or ("recv" in n and "nccl" in n)
            or ("peer" in n and "copy" in n)  # p2p copy
        )

    #  sample based profiler（only record args.profile_steps steps）
    prof = None
    prof_steps_target = max(0, int(args.profile_steps))
    prof_steps_done = 0
    prof_wall_ms = 0.0
    if prof_steps_target > 0:
        prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(wait=0, warmup=0, active=prof_steps_target, repeat=1),
            record_shapes=False, with_stack=False, profile_memory=False,
        )
        prof.__enter__()

    # 5) train loop
    step = 0
    global_bs = engine.train_batch_size()
    total_steps = math.ceil(args.max_examples / global_bs)
    assert args.max_examples > 0, "max_examples must be set"
    for _ in range(total_steps):
        t0 = time.time()
        loss = engine.train_batch()
        dt = time.time() - t0
        step += 1
        if prof is not None and prof_steps_done < prof_steps_target:
            prof_wall_ms += dt * 1000.0
            prof.step()
            prof_steps_done += 1
        # In pipeline parallelism the loss is only defined on the last stage.
        # Each call to train_batch() performs one optimizer step (after GAS micro-batches),
        # so we can log every N steps without checking a transient boundary flag.
        if engine.is_last_stage() and (step % args.log_every == 0):
            tokens = engine.train_batch_size() * args.seq_len
            tps = (tokens / dt) if dt > 0 else 0.0
            try:
                peak_mem_gb = (torch.cuda.max_memory_allocated() / (1024 ** 3)) if torch.cuda.is_available() else 0.0
            except Exception:
                peak_mem_gb = 0.0
            try:
                loss_value = round(float(loss.item()), 6) if loss is not None else None
            except Exception:
                loss_value = None
            log = {
                "step": step,
                "loss": loss_value,
                "tokens": int(tokens),
                "tokens_per_s": round(float(tps), 2),
                "step_time_ms": int(dt * 1000),
                "peak_mem_gb": round(float(peak_mem_gb), 2),
                "dataloader_idle_pct": None,
                "pack": False,
                "seq_len": args.seq_len,
            }
            print(log, flush=True)
            if torch.cuda.is_available():
                try:
                    torch.cuda.reset_peak_memory_stats()
                except Exception:
                    pass
    if prof is not None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        rank = dist.get_rank() if dist.is_initialized() else 0
        ev = prof.key_averages()  # 只统计 active 窗口

        def _cuda_us(e):
            # 兼容 CUDA/ROCm 和不同版本字段
            for attr in ("self_cuda_time_total", "cuda_time_total", "self_cuda_time", "device_time"):
                v = getattr(e, attr, 0.0)
                if v and float(v) > 0:
                    return float(v)
            return 0.0

        def _lname(s): return (s or "").lower()

        # 扩展通信关键字（NCCL/RCCL + c10d）
        def _is_comm_key(name: str) -> bool:
            n = _lname(name)
            return (
                "nccl" in n or "rccl" in n
                or "all_reduce" in n or "allreduce" in n
                or "reduce_scatter" in n or "all_gather" in n or "_all_gather_base" in n
                or "broadcast" in n
                or n.startswith("c10d::allreduce_") or n.startswith("c10d::all_gather")
                or n.startswith("c10d::reduce_scatter") or n.startswith("c10d::broadcast_")
            )

        # 扩展内存搬运关键字（ROCm 的 hip* 也算）
        def _is_mem_key(name: str) -> bool:
            n = _lname(name)
            return (
                "memcpy" in n or "memset" in n
                or "hipmemcpy" in n or "hipmemset" in n
                or "memcpyasync" in n or "hipmemcpyasync" in n or "hipmemsetasync" in n
            )

        # 避免把 ProfilerStep* 算进总 CUDA 时间
        filtered = [e for e in ev if not _lname(e.key).startswith("profilerstep")]
        cuda_us   = sum(_cuda_us(e) for e in filtered)
        comm_us   = sum(_cuda_us(e) for e in filtered if _is_comm_key(e.key))
        memcpy_us = sum(_cuda_us(e) for e in filtered if _is_mem_key(e.key))
        comp_us = max(0.0, cuda_us - comm_us - memcpy_us)

        # 粗略的“掩盖比例”估计：overlap_ms = max(0, comp+comm - step_wall)
        # 如果 overlap≈comm，则通信几乎被完全掩盖
        step_wall_ms = prof_wall_ms
        sum_ms = (comp_us + comm_us) / 1000.0
        overlap_ms = max(0.0, sum_ms - step_wall_ms)
        hidden_ratio = 0.0 if comm_us == 0 else min(1.0, overlap_ms / (comm_us / 1000.0))
        # if rank == 0:
        #     import pdb; pdb.set_trace()
        out = {
            "rank": rank,
            "profiled_steps": prof_steps_done,
            "T_comp_ms": round(comp_us/1000.0, 2),
            "T_comm_ms": round(comm_us/1000.0, 2),
            "T_mem_ms": round(memcpy_us/1000.0, 2),
            "wall_ms": round(step_wall_ms, 2),
            "est_hidden_comm_ratio": round(hidden_ratio, 3)
        }
        print({"profiling": out}, flush=True)
        torch.distributed.barrier()

        if args.profile_trace_dir:
            os.makedirs(args.profile_trace_dir, exist_ok=True)
            prof.export_chrome_trace(os.path.join(args.profile_trace_dir, f"trace_rank{rank}.json"))

        prof.__exit__(None, None, None)

def cleanup_dist():
    if dist.is_initialized():
        try:
            dist.barrier()
        except Exception:
            pass
        dist.destroy_process_group()


if __name__ == "__main__":
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
    try:
        main()
    finally:
        cleanup_dist()

