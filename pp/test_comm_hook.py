import os
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29505")

# 1) Install hook early (lazy loading will patch again after DS init)
from profiler import CommTimer
timer = CommTimer(enable=True)
timer.install()

# 2) Standard imports
import torch
import torch.distributed as dist
import deepspeed

rank = int(os.environ.get("RANK", "0"))
world = int(os.environ.get("WORLD_SIZE", os.environ.get("WORLD_SIZE", "1")))
local_rank = int(os.environ.get("LOCAL_RANK", "0"))

# 3) Initialize PG
if torch.cuda.is_available():
    torch.cuda.set_device(local_rank)
backend = "nccl" if torch.cuda.is_available() else "gloo"
if not dist.is_initialized():
    dist.init_process_group(backend=backend, init_method="env://", rank=rank, world_size=world)

# 4) Initialize DeepSpeed communication (this step makes deepspeed.comm fully functional)
deepspeed.comm.init_distributed(dist_backend=backend, dist_init_required=False)

# 5) Verify two necessary entries are wrapped
print(
    "HOOKED:",
    getattr(deepspeed.comm.all_reduce, "__wrapped_by_commtimer__", False),     # deepspeed.comm layer
    getattr(dist.all_reduce, "__wrapped_by_commtimer__", False),               # torch.distributed top layer
    " (cdb method detection ignored, many implementations are dynamically dispatched)",
    flush=True
)

# 6) Do one communication: two processes can see non-zero communication time
if world >= 2 and torch.cuda.is_available():
    x = torch.ones(1024, device="cuda") * (rank + 1)
    timer.begin_step()
    # Trigger through deepspeed.comm (common path in training)
    deepspeed.comm.all_reduce(x, op=dist.ReduceOp.SUM)
    t = timer.end_step()
    if t and rank == 0:
        print({"profiling_timer": {
            "world_size": world,
            "T_step_ms": round(t.step_ms, 3),
            "T_comm_ms": round(t.comm_ms, 3),
            "T_comp_ms": round(t.comp_ms, 3),
            "hidden_ratio_est": round(t.hidden_ratio_est, 3),
        }}, flush=True)
else:
    if rank == 0:
        print("SINGLE PROCESS/NO GPU ONLY VERIFIED HOOKED MARK; TO SEE T_COMM>0, PLEASE USE: torchrun --nproc_per_node=2 test_comm_hook.py", flush=True)
