# profiler.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable

import torch
import torch.distributed as dist


@dataclass
class StepTiming:
    step_ms: float
    comm_ms: float
    comp_ms: float
    hidden_ratio_est: float  # 1 - min(1, comm/step)


def should_enable_timer(step_index_1based: int, start_step: int, timer_steps: int) -> bool:
    """whether to enable timer on the given step. start_step skips the first K steps, timer_steps=0 means to record all steps."""
    if step_index_1based <= start_step:
        return False
    if timer_steps <= 0:
        return True
    return step_index_1based <= (start_step + timer_steps)


class CommTimer:
    """Use CUDA/HIP events to time collective operations; wrap train_batch() with step head and tail."""

    # These function names are wrapped in torch.distributed / deepspeed.comm / cdb.*
    TARGET_FUNCS = [
        "all_reduce", "all_gather_into_tensor", "reduce_scatter_tensor",
        "broadcast", "all_to_all_single", "send", "recv",
        # compatible with old names/aliases
        "all_gather", "reduce_scatter", "isend", "irecv", "barrier",
    ]

    def __init__(self, enable: bool = True):
        self.enable = bool(enable and torch.cuda.is_available())
        self._installed = False
        self._pairs: List[Tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self._step_evt_start: Optional[torch.cuda.Event] = None

    # ---------- Public API ----------

    def install(self) -> None:
        """Install hooks: can be called before or after distributed init."""
        if not self.enable or self._installed:
            return
        self._patch_all_namespaces()   # patch all namespaces first
        self._attach_lazy_hooks()      # intercept DS's init_distributed/set_backend, patch again after cdb is ready
        self._installed = True

    def begin_step(self) -> None:
        if not self.enable:
            return
        self._pairs.clear()
        torch.cuda.synchronize()
        self._step_evt_start = torch.cuda.Event(enable_timing=True)
        self._step_evt_start.record(torch.cuda.current_stream())

    def end_step(self) -> Optional[StepTiming]:
        if not self.enable or self._step_evt_start is None:
            return None

        # record step end
        step_end = torch.cuda.Event(enable_timing=True)
        step_end.record(torch.cuda.current_stream())
        torch.cuda.synchronize()

        # record step duration
        step_ms = float(self._step_evt_start.elapsed_time(step_end))

        # convert each collective's start and end events to [t0, t1] interval relative to step start
        intervals = []
        for s, e in self._pairs:
            try:
                t0 = float(self._step_evt_start.elapsed_time(s))
                t1 = float(self._step_evt_start.elapsed_time(e))
                if t1 < t0:
                    t0, t1 = t1, t0
                # clamp to [0, step_ms], and ignore very short/abnormal intervals
                t0 = max(0.0, min(t0, step_ms))
                t1 = max(0.0, min(t1, step_ms))
                if t1 - t0 > 1e-3:
                    intervals.append((t0, t1))
            except Exception:
                # skip events that cannot be measured
                pass

        # merge intervals (line segment union), get communication coverage duration
        intervals.sort()
        comm_union = 0.0
        cur_s = cur_e = None
        for s0, e0 in intervals:
            if cur_s is None:
                cur_s, cur_e = s0, e0
            elif s0 <= cur_e:
                cur_e = max(cur_e, e0)
            else:
                comm_union += (cur_e - cur_s)
                cur_s, cur_e = s0, e0
        if cur_s is not None:
            comm_union += (cur_e - cur_s)

        # fallback: if none are merged, return "sum and truncate"
        if comm_union <= 0.0 and self._pairs:
            raw_sum = 0.0
            for s, e in self._pairs:
                try:
                    raw_sum += float(s.elapsed_time(e))
                except Exception:
                    pass
            comm_union = min(raw_sum, step_ms)

        comm_ms = float(comm_union)
        comp_ms = max(0.0, step_ms - comm_ms)
        hidden = 0.0 if step_ms <= 0 else max(0.0, 1.0 - min(1.0, comm_ms / step_ms))

        self._step_evt_start = None
        self._pairs.clear()
        return StepTiming(step_ms=step_ms, comm_ms=comm_ms, comp_ms=comp_ms, hidden_ratio_est=hidden)

    @staticmethod
    def rank() -> int:
        try:
            return dist.get_rank() if dist.is_initialized() else 0
        except Exception:
            return 0

    # ---------- Flatten: below are internal tools, no nesting ----------

    def _wrap_collective_fn(self, fn: Callable) -> Callable:
        """Module function/free function wrapper: record events on the same stream before and after calling, without changing the function signature, no synchronization."""
        def _inner(*args, **kwargs):
            if not self.enable or not torch.cuda.is_available():
                return fn(*args, **kwargs)
            dev = torch.cuda.current_device()
            stream = torch.cuda.current_stream(dev)

            s = torch.cuda.Event(enable_timing=True)
            s.record(stream)
            try:
                return fn(*args, **kwargs)
            finally:
                e = torch.cuda.Event(enable_timing=True)
                e.record(stream)
                self._pairs.append((s, e))

        _inner.__wrapped_by_commtimer__ = True
        return _inner

    def _wrap_class_method(self, cls: type, name: str) -> None:
        """Class method wrapper: patch cdb.__class__ to avoid instance binding issues."""
        if not hasattr(cls, name):
            return
        method = getattr(cls, name)
        if not callable(method) or getattr(method, "__wrapped_by_commtimer__", False):
            return

        def wrapped(inst, *args, **kwargs):
            if not self.enable or not torch.cuda.is_available():
                return method(inst, *args, **kwargs)
            dev = torch.cuda.current_device()
            stream = torch.cuda.current_stream(dev)
            s = torch.cuda.Event(enable_timing=True); s.record(stream)
            try:
                return method(inst, *args, **kwargs)
            finally:
                e = torch.cuda.Event(enable_timing=True); e.record(stream)
                self._pairs.append((s, e))
        wrapped.__wrapped_by_commtimer__ = True
        setattr(cls, name, wrapped)

    def _wrap_module_space(self, space, names: List[str]) -> None:
        """Patch callable members in a given namespace (module or object)."""
        if space is None:
            return
        for n in names:
            if not hasattr(space, n):
                continue
            fn = getattr(space, n)
            if callable(fn) and not getattr(fn, "__wrapped_by_commtimer__", False):
                setattr(space, n, self._wrap_collective_fn(fn))

    def _patch_all_namespaces(self) -> None:
        """Patch all torch.distributed, deepspeed.comm, and ready cdb."""
        # 1) torch.distributed 顶层与 distributed_c10d
        try:
            import torch.distributed as tdist
            try:
                self._wrap_module_space(tdist.distributed_c10d, self.TARGET_FUNCS)
            except Exception:
                pass
            self._wrap_module_space(tdist, self.TARGET_FUNCS)
        except Exception:
            pass

        # 2) deepspeed.comm module functions
        ds_comm = None
        try:
            import deepspeed
            ds_comm = deepspeed.comm
            self._wrap_module_space(ds_comm, self.TARGET_FUNCS)
        except Exception:
            ds_comm = None

        # 3) if cdb is ready, patch its "class methods" (fallback)
        try:
            if ds_comm is not None and getattr(ds_comm, "cdb", None) is not None:
                cdb = ds_comm.cdb
                self._wrap_module_space(cdb, self.TARGET_FUNCS)  # instance-level fallback (most implementations are无效，但无伤大雅）
                for n in self.TARGET_FUNCS:
                    self._wrap_class_method(cdb.__class__, n)
        except Exception:
            pass

    def _attach_lazy_hooks(self) -> None:
        """Intercept DS's init_distributed/set_backend, patch again after cdb is ready."""
        try:
            import deepspeed
            ds_comm = deepspeed.comm
        except Exception:
            return

        # wrap one layer: run _patch_all_namespaces after the original function returns
        def _wrap_after(name: str):
            if not hasattr(ds_comm, name):
                return
            orig = getattr(ds_comm, name)
            if getattr(orig, "__ct_wrapped__", False):
                return

            def wrapper(*a, **k):
                r = orig(*a, **k)
                try:
                    self._patch_all_namespaces()
                finally:
                    return r
            wrapper.__ct_wrapped__ = True
            setattr(ds_comm, name, wrapper)

        _wrap_after("init_distributed")
        _wrap_after("set_backend")
