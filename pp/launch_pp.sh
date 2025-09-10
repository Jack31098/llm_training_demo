#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0,1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export OMP_NUM_THREADS=24
export QWEN_SDP_FORCE=math

#  --fp16 \
#  --bf16 \
deepspeed --num_gpus=2 pp/pp_train.py \
  --model_name_or_path ./models/qwen3-0.6b-base \
  --train_json data/sft_en.jsonl \
  --seq_len 512 \
  --bf16 \
  --micro_batch_size 1 \
  --grad_accum_steps 1 \
  --epochs 1 \
  --log_every 8 \
  --deepspeed_config pp/ds_pp_2gpus.json


