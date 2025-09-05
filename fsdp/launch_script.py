import os
import sys
from typing import List


def _set_default_env_for_single_process() -> None:
    # Torch distributed single-process defaults
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")


def main(extra_args: List[str] | None = None) -> None:
    # Match your bash script's environment and arguments
    os.environ.setdefault("QWEN_DEBUG_ATTENTION", "0")
    # os.environ.setdefault("QWEN_DEBUG_NAN", "1")
    os.environ.setdefault("QWEN_SDP_FORCE", "math")
    # os.environ.setdefault("DEBUG_ATTN", "1")
    os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")

    _set_default_env_for_single_process()

    # Build the same CLI args as launch_fsdp.sh (single GPU, no FSDP)
    args_list: List[str] = [
        "--model_name_or_path", "llm_demo/models/qwen3-0.6b-base",
        "--train_json", "llm_demo/data/sft_en.jsonl",
        "--seq_len", "512",
        "--lr", "0.0",
        "--wd", "0.0",
        "--no_fsdp",
        "--debug_autograd",
        "--sanitize_grads",
        "--micro_batch_size", "1",
        "--grad_accum_steps", "1",
        "--max_steps", "10",
        "--log_every", "5",
        "--output_dir", "llm_demo/runs/fsdp_s1024_ga50",
    ]

    if extra_args:
        args_list.extend(extra_args)

    # Import here to avoid side effects before env is set
    from train_fsdp import parse_args, train
    # Temporarily set sys.argv to parse with the project’s parser
    _orig_argv = sys.argv
    try:
        sys.argv = [sys.argv[0]] + args_list
        args = parse_args()
    finally:
        sys.argv = _orig_argv

    train(args)


if __name__ == "__main__":
    # Allow passing overrides via the IDE run config, e.g.:
    # python launch_script.py --seq_len 256 --attn_impl eager
    main(sys.argv[1:])


