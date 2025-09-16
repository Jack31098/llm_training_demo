#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0,1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export OMP_NUM_THREADS=24
export QWEN_SDP_FORCE=math
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

#  --fp16 \
#  --bf16 \
deepspeed --num_gpus=2 pp/pp_train.py \
  --model_name_or_path ./models/qwen3-0.6b-base \
  --train_json data/sft_en.jsonl \
  --seq_len 1024 \
  --bf16 \
  --max_examples 128 \
  --log_every 2 \
  --deepspeed_config pp/ds_pp_2gpus.json


