#!/usr/bin/env python
# Minimal SFT training skeleton with FSDP  on ROCm
# - Logs per-step metrics: tokens/s, step time, peak memory, DataLoader idle%
# - Optional per-sample loss debug (without sacrificing packing efficiency)
# - Sequence packing to improve utilization (concat samples until seq_len)
# - BF16 + activation checkpointing + grad accumulation
# - FSDP with size-based auto-wrap; safe defaults for AMD MI100 (gfx908)
# NOTE: This is a single-file demo. For real use, split into modules.

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from functools import partial

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, FullStateDictConfig
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.distributed.fsdp import MixedPrecision
from torch.amp import GradScaler, autocast
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------------------
# Dataset & Packing Utilities
# ---------------------------
class JsonlSFTDataset(Dataset):
    """Assumes JSONL with fields {"input": str, "target": str}.
    You can adapt field names via CLI.
    """
    def __init__(self, path: str, input_key: str = "input", target_key: str = "target"):
        self.samples: List[Dict[str, str]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                self.samples.append({
                    "input": obj.get(input_key, ""),
                    "target": obj.get(target_key, ""),
                })
        if len(self.samples) == 0:
            raise ValueError(f"No samples found in {path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

@dataclass
class PackedBatch:
    input_ids: torch.Tensor           # [B, S]
    attention_mask: torch.Tensor      # [B, S]
    labels: torch.Tensor              # [B, S], -100 for padding
    # For optional per-sample loss debug: maps packed positions to original sample ids
    # Each element: (packed_row_idx, start_pos, end_pos_exclusive, original_sample_index)
    segments: List[Tuple[int, int, int, int]]

class Packer:
    def __init__(self, tokenizer: AutoTokenizer, seq_len: int, eos_token_id: int):
        self.tok = tokenizer
        self.S = seq_len
        self.eos = eos_token_id

    def tokenize(self, sample: Dict[str, str]) -> List[int]:
        # Simple prompt format: [input]\n[answer]\n + EOS on target end
        text = sample["input"].rstrip() + "\n" + sample["target"].rstrip() + self.tok.eos_token
        ids = self.tok(text, add_special_tokens=False)["input_ids"]
        return ids

    def collate_pack(self, batch_samples: List[Dict[str, str]], b_micro: int) -> PackedBatch:
        # Greedy pack: fill each row with as many samples as fit (with EOS boundaries)
        B, S = b_micro, self.S
        input_ids = torch.full((B, S), fill_value=self.tok.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((B, S), dtype=torch.long)
        labels = torch.full((B, S), fill_value=-100, dtype=torch.long)
        segments: List[Tuple[int,int,int,int]] = []

        row = 0
        col = 0
        for idx, sample in enumerate(batch_samples):
            ids = self.tokenize(sample)
            pos = 0
            while pos < len(ids):
                if row >= B:
                    break
                space = S - col
                take = min(space, len(ids) - pos)
                if take > 0:
                    input_ids[row, col:col+take] = torch.tensor(ids[pos:pos+take], dtype=torch.long)
                    attention_mask[row, col:col+take] = 1
                    labels[row, col:col+take] = input_ids[row, col:col+take]
                    segments.append((row, col, col+take, idx))
                    col += take
                    pos += take
                if col == S:
                    row += 1
                    col = 0
            if row >= B:
                break
        return PackedBatch(input_ids, attention_mask, labels, segments)

    def collate_no_pack(self, batch_samples: List[Dict[str, str]], b_micro: int) -> PackedBatch:
        # One sample per row; truncate/pad to S
        B, S = b_micro, self.S
        input_ids = torch.full((B, S), fill_value=self.tok.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((B, S), dtype=torch.long)
        labels = torch.full((B, S), fill_value=-100, dtype=torch.long)
        segments: List[Tuple[int,int,int,int]] = []
        for i in range(B):
            if i >= len(batch_samples):
                break
            ids = self.tokenize(batch_samples[i])[:S]
            L = len(ids)
            input_ids[i, :L] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, :L] = 1
            labels[i, :L] = input_ids[i, :L]
            segments.append((i, 0, L, i))
        return PackedBatch(input_ids, attention_mask, labels, segments)

# ---------------------------
# Training Utilities
# ---------------------------

def setup_dist():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", device_id=device)  # ROCm uses RCCL under the hood
    
    return local_rank, device


def all_reduce_mean(x: torch.Tensor) -> torch.Tensor:
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    x /= dist.get_world_size()
    return x


def save_fsdp_full(model: FSDP, out_dir: str, step: int):
    out_dir = Path(out_dir)
    if dist.get_rank() == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
    # Ensure directory exists before any rank proceeds
    dist.barrier()
    # All ranks must participate in FULL_STATE_DICT gathering even with rank0_only=True
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):
        state = model.state_dict()
    if dist.get_rank() == 0:
        torch.save(state, out_dir / f"checkpoint_step{step}.pt")
    # Synchronize after saving to avoid shutdown races
    dist.barrier()


def save_plain_full(model: torch.nn.Module, out_dir: str, step: int):
    out_dir = Path(out_dir)
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()
    state = model.state_dict()
    if rank == 0:
        torch.save(state, out_dir / f"checkpoint_step{step}.pt")
    if dist.is_initialized():
        dist.barrier()

# ---------------------------
# Main Training Loop
# ---------------------------

def nan_inf_hook(name):
    def hook(module, input, output):
        if not isinstance(output, torch.Tensor):
            return
        if torch.isnan(output).any() or torch.isinf(output).any():
            print(f"NaN/Inf detected in {name}")
    return hook

def train(args):
    rank, device = setup_dist()
    is_master = rank == 0

    if is_master:
        os.makedirs(args.output_dir, exist_ok=True)
        (Path(args.output_dir) / "metrics").mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Load model
    # Choose dtype from flags (default FP32; enable BF16/FP16 only when flag is provided)
    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
    print("passin dtype", dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        dtype=dtype,
    )
    model_gradient_ckpt_supported = hasattr(model, "gradient_checkpointing_enable")
    if args.use_activation_checkpointing and model_gradient_ckpt_supported:
        model.gradient_checkpointing_enable()
    model.config.use_cache = False
    # model.config.attn_implementation = "eager"
    mp_policy = None

    # FSDP config (optional)
    use_fsdp = dist.get_world_size() > 1
    if use_fsdp:
        wrap_policy = partial(size_based_auto_wrap_policy, min_num_params=int(args.fsdp_wrap_min_params))
        # Set FSDP mixed precision to match chosen dtype
        # if dtype == torch.bfloat16:
        #     mp_policy = MixedPrecision(param_dtype=torch.float32, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16)
        # For FP16, disable mixed precision to avoid conflicts with ShardedGradScaler
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        model = FSDP(model, auto_wrap_policy=wrap_policy, mixed_precision=mp_policy, device_id=local_rank)
    else:
        model = model.to(device)

    # Attention implementation selectable
    model.config.attn_implementation = getattr(args, "attn_impl")
    
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_math_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    # print(torch.backends.cuda.flash_sdp_enabled())
    # print(torch.backends.cuda.mem_efficient_sdp_enabled())
    # print(torch.backends.cuda.math_sdp_enabled())


    # Optional per-layer NaN/Inf forward detection (rank0 only)
    nan_hook_handles: List[Any] = []
    if args.debug_nan and is_master:
        for name, module in model.named_modules():
            if "layer" in name or "attn" in name:  # Focus on key layers
                handle = module.register_forward_hook(nan_inf_hook(name))
                nan_hook_handles.append(handle)

    # Optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    # AMP scaler only for FP16 training
    if dtype == torch.float16:
        if use_fsdp:
            scaler = ShardedGradScaler()
        else:
            scaler = GradScaler('cuda')
    else:
        scaler = None

    # Data
    ds = JsonlSFTDataset(args.train_json, input_key=args.input_key, target_key=args.target_key)
    packer = Packer(tok, seq_len=args.seq_len, eos_token_id=tok.eos_token_id)

    def make_batch(samples: List[Dict[str,str]]) -> PackedBatch:
        if args.pack_sequences:
            return packer.collate_pack(samples, args.micro_batch_size)
        else:
            return packer.collate_no_pack(samples, args.micro_batch_size)

    # Simple sampler: shard by rank
    indices = list(range(len(ds)))
    random.seed(args.seed + rank)
    random.shuffle(indices)

    steps = 0
    model.train()

    # For DataLoader idle time, we pull batches manually
    it = 0
    while steps < args.max_steps:
        # --------------- data fetch ---------------
        t0 = time.time()
        batch_samples = []
        while len(batch_samples) < args.micro_batch_size:
            if it >= len(indices):
                # reshuffle for next epoch
                random.shuffle(indices)
                it = 0
            batch_samples.append(ds[indices[it]])
            it += 1
        packed = make_batch(batch_samples)
        fetch_time = time.time() - t0

        # Move to device
        input_ids = packed.input_ids.to(device, non_blocking=True)
        attention_mask = packed.attention_mask.to(device, non_blocking=True)
        labels = packed.labels.to(device, non_blocking=True)

        # --------------- forward+backward (GA) ---------------
        torch.cuda.reset_peak_memory_stats(device)
        step_compute_t0 = time.time()
        total_tokens_this_step = 0
        total_loss = 0.0

        # Count tokens (labels != -100)
        total_tokens_this_step += int((labels != -100).sum().item())

        def compute_safe_ce_loss():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # [B,S,V]
            # Debug logit stats (rank0 only)
            if is_master and (steps % args.log_every == 0 or steps <= 5):
                logit_mean = logits.mean().item()
                logit_min = logits.min().item()
                logit_max = logits.max().item()
                print(json.dumps({"debug_logits": {"mean": round(logit_mean, 2), "min": round(logit_min, 2), "max": round(logit_max, 2)}}))
            # Sanitize logits to avoid NaNs/Infs propagating into CE
            logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
            # Shift for causal LM
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            # Flatten
            vocab = shift_logits.size(-1)
            flat_logits = shift_logits.view(-1, vocab).float()  # compute CE in FP32 for stability
            flat_labels = shift_labels.view(-1)
            valid_mask = flat_labels != -100
            num_valid = int(valid_mask.sum().item())
            if num_valid == 0:
                return torch.tensor(0.0, device=device, dtype=torch.float32)
            # CE on valid positions only
            loss_sum = F.cross_entropy(flat_logits[valid_mask], flat_labels[valid_mask], reduction="sum")
            return loss_sum / num_valid

        autocast_dtype = torch.bfloat16 if dtype == torch.bfloat16 else (torch.float16 if dtype == torch.float16 else None)
        if autocast_dtype is not None:
            with autocast(dtype=autocast_dtype, device_type='cuda'):
                # loss = compute_safe_ce_loss()
                loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
        else:
            loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
            # loss = compute_safe_ce_loss()

        loss = loss / args.grad_accum_steps
        # Guard against non-finite loss to avoid poisoning optimizer state
        if not torch.isfinite(loss):
            if is_master:
                # Minimal diagnostics
                num_valid_tokens = int((labels[:, 1:] != -100).sum().item())
                print(json.dumps({"warning": "non_finite_loss", "step": int(steps+1), "valid_tokens": num_valid_tokens}))
            opt.zero_grad(set_to_none=True)
            # Skip this micro-step and continue
            continue

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Optional: sanitize gradients to avoid NaN/Inf propagation
        if getattr(args, "sanitize_grads", False):
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                    p.grad.clamp_(min=-1e3, max=1e3)
        # Check gradient finiteness and skip bad micro-steps to protect weights
        bad_grad = False
        for p in model.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                bad_grad = True
                break
        if bad_grad:
            if is_master:
                print(json.dumps({"warning": "non_finite_grad", "step": int(steps+1)}))
            opt.zero_grad(set_to_none=True)
            continue
        total_loss += loss.detach().float().item()

        if (steps + 1) % args.grad_accum_steps == 0:
            if scaler is not None:
                if args.max_grad_norm > 0:
                    scaler.unscale_(opt)
                    clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            else:
                if args.max_grad_norm > 0:
                    clip_grad_norm_(model.parameters(), args.max_grad_norm)
                opt.step()
                opt.zero_grad(set_to_none=True)

        # --------------- metrics ---------------
        step_compute = time.time() - step_compute_t0
        # all-reduce metrics
        t_toks = torch.tensor([total_tokens_this_step], device=device, dtype=torch.float64)
        t_loss = torch.tensor([total_loss * args.grad_accum_steps], device=device, dtype=torch.float64)
        t_step = torch.tensor([step_compute], device=device, dtype=torch.float64)
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(t_toks, op=dist.ReduceOp.SUM)
            dist.all_reduce(t_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(t_step, op=dist.ReduceOp.MAX)
        world = dist.get_world_size() if dist.is_initialized() else 1
        global_tokens = int(t_toks.item())
        global_loss = float(t_loss.item() / world)
        step_time = float(t_step.item())
        tokens_per_s = global_tokens / max(step_time, 1e-6)
        peak_mem_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
        data_idle_pct = 100.0 * fetch_time / max(fetch_time + step_time, 1e-6)

        steps += 1

        if is_master and (steps % args.log_every == 0 or steps <= 5):
            log = {
                "step": steps,
                "loss": round(global_loss, 6),
                "tokens": global_tokens,
                "tokens_per_s": round(tokens_per_s, 2),
                "step_time_ms": int(step_time * 1000),
                "peak_mem_gb": round(float(peak_mem_gb), 2),
                "dataloader_idle_pct": round(data_idle_pct, 1),
                "pack": bool(args.pack_sequences),
                "seq_len": args.seq_len,
                "micro_batch_size": args.micro_batch_size,
                "grad_accum_steps": args.grad_accum_steps,
            }
            print(json.dumps(log, ensure_ascii=False))
            with open(Path(args.output_dir)/"metrics"/"train_log.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(log) + "\n")

        # optional: per-sample loss debug (costly; run sparsely)
        if is_master and args.debug_per_sample_loss and (steps % args.debug_every == 0):
            with torch.no_grad():
                # compute token-wise NLL and aggregate per original sample id
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits  # [B,S,V]
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                # cross-entropy per token
                per_tok_loss = torch.nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1), reduction="none"
                ).view(shift_labels.size())  # [B,S-1]
                # pad to [B,S] for indexing simplicity
                per_tok_loss = torch.nn.functional.pad(per_tok_loss, (1,0))  # add one dummy at start
                # aggregate over segments
                per_sample: Dict[int, float] = {}
                per_sample_tok: Dict[int, int] = {}
                for (r, s, e, sid) in packed.segments:
                    # avoid counting -100 labels
                    mask = (labels[r, s:e] != -100)
                    if mask.any():
                        l = per_tok_loss[r, s:e][mask].sum().item()
                        n = int(mask.sum().item())
                        per_sample[sid] = per_sample.get(sid, 0.0) + l
                        per_sample_tok[sid] = per_sample_tok.get(sid, 0) + n
                debug_rows = []
                for sid, l in per_sample.items():
                    n = max(per_sample_tok.get(sid, 1), 1)
                    debug_rows.append({"sample_idx": int(sid), "loss": l / n, "tokens": n})
                dbg_path = Path(args.output_dir)/"metrics"/f"per_sample_step{steps}.json"
                with open(dbg_path, "w", encoding="utf-8") as f:
                    json.dump(debug_rows, f, ensure_ascii=False, indent=2)

        if args.save_ckpt_every > 0 and steps % args.save_ckpt_every == 0:
            if use_fsdp:
                save_fsdp_full(model, args.output_dir, steps)
            else:
                save_plain_full(model, args.output_dir, steps)

    # final checkpoint
    if use_fsdp:
        save_fsdp_full(model, args.output_dir, steps)
    else:
        save_plain_full(model, args.output_dir, steps)
    # Cleanly shutdown the process group to avoid resource leak warnings
    if dist.is_initialized():
        dist.destroy_process_group()


def parse_args():
    p = argparse.ArgumentParser()
    # data/model
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
    p.add_argument("--pack_sequences", action="store_true")
    p.add_argument("--micro_batch_size", type=int, default=1)
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--lr", type=float, default=0)
    p.add_argument("--wd", type=float, default=0.1)
    p.add_argument("--max_grad_norm", type=float, default=0.01)
    p.add_argument("--max_steps", type=int, default=100)
    # fsdp
    p.add_argument("--fsdp_wrap_min_params", type=float, default=1e7, help="Auto-wrap modules above this param count")
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
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
