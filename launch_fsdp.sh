cd "$(dirname "$0")"
export HIP_VISIBLE_DEVICES=0,1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export OMP_NUM_THREADS=24

#  --pack_sequences \
#     --bf16 \
torchrun --nproc_per_node=2 train_fsdp.py \
  --model_name_or_path ./models/qwen3-0.6b-base \
  --train_json data/sft_en.jsonl \
  --seq_len 512 --lr 1e-5 --wd 0.1 \
  --bf16 \
  --use_activation_checkpointing \
  --attn_impl eager \
  --micro_batch_size 16 --grad_accum_steps 2 --max_steps 48 \
  --log_every 8 --output_dir runs/fsdp_s1024_ga50
