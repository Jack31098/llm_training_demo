Goal

Demonstrate supervised fine-tuning (SFT) of a Qwen-class 8B decoder using FSDP and FSDP + 2-way Pipeline Parallel (PP) on an AMD EPYC 7403 + 2× MI100 (gfx908) node.
We will:

establish a FSDP baseline (BF16 + activation checkpointing + token packing),

add 2-way PP (1F1B) when memory-bound,

and compare throughput under different gradient accumulation (GA) settings that push the effective tokens per step to approximately T_eff ≈ 100k.

Definitions

N: world size (number of GPUs). Here N = 2.
B_micro: per-device micro-batch size (tokens are counted separately).
GA: gradient accumulation steps.
Global batch: B_global = B_micro × GA × N.
Sequence length: S (tokens per sequence).
Effective tokens per step: T_eff = S × B_global.
Example that hits ~100k: S=1024, B_micro=1, GA=50, N=2 → T_eff = 1024 × 1 × 50 × 2 = 102,400.


Hardware & Software
Node: AMD EPYC 7003 (PCIe)
GPUs: 2× MI100 (gfx908)
OS/Drivers: UBUNTU24.04 lts/ROCm 6.4

Python: 3.12
PyTorch: 2.8 (ROCm build), torch.distributed backend = "nccl" (RCCL)
Optional: SDPA’s FlashAttention backend on ROCm/gfx908 (feature-gated)
Tokenizer/Models: Qwen3-class 8B decoder (HF style)