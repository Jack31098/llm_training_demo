## Key Results of deepspeed PP training
#### Node:AMD EPYC 7003 (PCIe) GPUs: 2× MI100 (gfx908) OS/Drivers: UBUNTU24.04 lts/ROCm 6.4
#### Benchmarked with Qwen3 0.6B, seq_len=1024, over 32 steps, the readings are avg

#### Legend: 
`step_ms` avg over 32 steps (excl. data); 
`comm_ms` RCCL avg via `custom profiler`;  
`toks/s` global tokens/s (all GPUs); 
`peak_mem(G)` per-GPU 

| micro_bsz | GA | ckpt | dtype | step_ms | comm_ms | toks/s | peak_mem(G) |
|----------:|---:|:----:|:----:|-------:|--------:|------:|------------:|
| 1 | 1 | F | FP32 | 477 | 226 | 2239 | 9.1 |
| 1 | 1 | T | BF16 | 596 | 294 | 1790 | 8.41 |
| 4 | 8 | T | BF16 | 5998 | 679 | 5485 | 13.6 |
| 4 | 8 | T | FP32 | 7184 | 681 | 4574 | 17 |
| 4 | 8 | F | FP32 | 5404 | 517 | 6085 | 27.3 |
| 8 | 1 | T | BF16 | 2187 | 988 | 3775 | 17.8 |
| 8 | 4 | T | BF16 | 5669 | 1004 | 5800 | 20.16 |
| 8 | 8 | T | BF16 | 10418 | 1094 | 6303 | 20.16 |
| 8 | 8 | T | FP32 | 14264 | 1618 | 5065 | 28.35 |

# Key observations
PP: DeepSpeed 2-way pipeline with 1F1B (no ZeRO-2/3 in this config).
#### Observation 1. On this setup, DS PP > native FSDP in throughput at GA≥8, while memory grows due to FP32 master weights.
#### Observation 2. DS PP + ZeRO-2/3 was not used here due to incompatibilities with deepspeed version 0.17.1; results may differ on other versions.
#### Observation 3. DS PP keeps FP32 master weights with FP16 optimizer, which increases memory footprint per layer.


## Known limitations (this repo, this setup)

SDPA (memory-efficient) on ROCm/gfx908: no GQA support in PyTorch 2.8 ROCm; emulation via tiling K/V negates gains.

FlashAttention via SDPA backend: non-None attention masks unsupported on this backend for gfx908.

Fallback: use SDPA math or eager attention where above constraints apply.
We will enable masked FA and GQA-aware MEA once upstream supports them on ROCm gfx908.


## Goal

Demonstrate supervised fine-tuning (SFT) of a Qwen3 base 0.6B decoder using FSDP and 2-way Pipeline Parallel (PP) on an AMD EPYC 7403 + 2× MI100 (gfx908) node.
We will:

establish a FSDP baseline (BF16 + activation checkpointing + token packing),

add 2-way PP (1F1B) when memory-bound,

and compare throughput under different gradient accumulation (GA) settings that push the effective tokens per step to approximately T_eff ≈ 100k.




## Definitions

N: world size (number of GPUs). Here N = 2.
B_micro: per-device micro-batch size (tokens are counted separately).
GA: gradient accumulation steps.
Global batch: B_global = B_micro × GA × N.
Sequence length: S (tokens per sequence).
Effective tokens per step: T_eff = S × B_global.
Example that hits ~100k: S=1024, B_micro=1, GA=50, N=2 → T_eff = 1024 × 1 × 50 × 2 = 102,400.


## Hardware & Software
Node: AMD EPYC 7003 (PCIe)
GPUs: 2× MI100 (gfx908)
OS/Drivers: UBUNTU24.04 lts/ROCm 6.4

Python: 3.12
PyTorch: 2.8 (ROCm build), torch.distributed backend = "nccl" (RCCL)
Optional: SDPA’s FlashAttention backend on ROCm/gfx908 (feature-gated)
Tokenizer/Models: Qwen3-base 0.6B decoder (HF style)


## Setup

- Create and activate a Python 3.12 environment.
- Install ROCm PyTorch 2.8 per official instructions (ROCm wheels).
- Install project deps:

pip install -r llm_demo/requirements.txt
#### or as a package (adds console script llm-demo-fsdp)
pip install -e llm_demo


#### Data & models

- Place a HF-compatible Qwen3 model under llm_demo/models/ (or use a path).
- Training data JSONL example provided at llm_demo/data/sft_en.jsonl.


#### Run: FSDP 

Uses the console script wrapper to mirror env from fsdp/launch_script.py
bash fsdp/launch_fsdp.sh

This calls train_fsdp.py with defaults set in the launcher (edit as needed).


#### Run: 2-way pipeline parallel (DeepSpeed, 2 GPUs)

bash pp/launch_pp.sh


Notes

- On ROCm, torch.cuda APIs are used; NCCL maps to RCCL.
- For attention backend: train_fsdp.py sets SDPA math backend to avoid unsupported FA+mask on gfx908.
- Adjust GA to target effective tokens T_eff = S × B_micro × GA × N.

