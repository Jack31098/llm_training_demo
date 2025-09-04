cd "$(dirname "$0")"
export HIP_VISIBLE_DEVICES=0,1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export OMP_NUM_THREADS=24

torchrun --nproc_per_node=2 train_fsdp.py \
  --model_name_or_path ./models/qwen3-0.6b-base \
  --train_json data/sft_en.jsonl \
  --bf16 \
  --seq_len 512 --lr 0.0 --wd 0.0 \
  --use_activation_checkpointing \
  --pack_sequences \
  --attn_impl sdpa \
  --micro_batch_size 1 --grad_accum_steps 1 --max_steps 50 \
  --log_every 1 --output_dir runs/fsdp_s1024_ga50
