Current status & limitations

AOTriton FlashAttention (PyTorch 2.8, ROCm): non-None attn_mask is not supported through the SDPA integration. Our training data pipeline requires attention masks, so FA cannot be used on masked batches.

SDPA memory-efficient attention (MEA): does not support GQA (num_heads_q != num_heads_kv). Emulating GQA by tiling/broadcasting K/V inside the kernel introduces extra copies and negates the expected speed/VRAM gains.

Fallback in this repo: training currently uses either the eager attention path or SDPA with the math backend.

We’ll update this section once kernel support for masked FA and GQA-aware MEA is available.


Goal

Demonstrate supervised fine-tuning (SFT) of a Qwen-class 8B decoder using FSDP and FSDP + 2-way Pipeline Parallel (PP) on an AMD EPYC 7403 + 2× MI100 (gfx908) node.
We will:

establish a FSDP baseline (BF16 + activation checkpointing + token packing),

add 2-way PP (1F1B) when memory-bound,

and compare throughput under different gradient accumulation (GA) settings that push the effective tokens per step to approximately T_eff ≈ 100k.

Key Results
| m (micro-batches) | step\_ms ↓ | toks/s ↑ | peak\_mem (GB) ↓ |  ckpt  | bf16 |  GA |
| ----------------: | ---------: | -------: | ---------------: | :----: | :--: | --: |
|                 1 |          … |        … |                … |   off  |  …   |  …  |
|                 1 |          … |        … |                … |   on   |  …   |  …  |
|                 8 |          … |        … |                … | **on** |  …   |  …  |
|                16 |          … |        … |                … |   on   |  …   |  …  |



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


Setup

- Create and activate a Python 3.12 environment.
- Install ROCm PyTorch 2.8 per official instructions (ROCm wheels).
- Install project deps:

pip install -r llm_demo/requirements.txt
# or as a package (adds console script llm-demo-fsdp)
pip install -e llm_demo


Data & models

- Place a HF-compatible Qwen3 model under llm_demo/models/ (or use a path).
- Training data JSONL example provided at llm_demo/data/sft_en.jsonl.


Run: FSDP single GPU (baseline skeleton)

# Uses the console script wrapper to mirror env from fsdp/launch_script.py
llm-demo-fsdp --seq_len 512 --attn_impl sdpa

Or directly:

python -m llm_demo.fsdp.launch_script --seq_len 512 --attn_impl sdpa

This calls train_fsdp.py with defaults set in the launcher (edit as needed).


Run: DDP + PowerSGD (2 GPUs)

torchrun --standalone --nproc_per_node=2 llm_demo/train_powerSGD.py \
  --model_name_or_path llm_demo/models/qwen3-0.6b-base \
  --train_json llm_demo/data/sft_en.jsonl \
  --seq_len 1024 --micro_batch_size 1 --grad_accum_steps 50 \
  --use_powersgd --psgd_rank 1 --output_dir llm_demo/runs/ddp_demo


Run: 2-way pipeline parallel (DeepSpeed, 2 GPUs)

deepspeed --num_gpus=2 llm_demo/pp/pp_train.py \
  --deepspeed_config llm_demo/pp/ds_pp_2gpus.json \
  --model_name_or_path llm_demo/models/qwen3-0.6b-base \
  --train_json llm_demo/data/sft_en.jsonl \
  --seq_len 1024 --log_every 10


Notes

- On ROCm, torch.cuda APIs are used; NCCL maps to RCCL.
- For attention backend: train_fsdp.py sets SDPA math backend to avoid unsupported FA+mask on gfx908.
- Adjust GA to target effective tokens T_eff = S × B_micro × GA × N.

