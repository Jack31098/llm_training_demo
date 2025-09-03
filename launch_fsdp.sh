torchrun --nproc_per_node=2 train_fsdp.py \
  --model_name_or_path ./models/Qwen3-0.6B-Base \
  --train_json data/sft.jsonl \
  --seq_len 1024 --bf16 --use_activation_checkpointing --pack_sequences \
  --micro_batch_size 1 --grad_accum_steps 50 \
  --log_every 20 --output_dir runs/fsdp_s1024_ga50
