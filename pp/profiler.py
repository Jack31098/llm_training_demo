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
    """是否在给定 step 开启计时窗口。start_step 跳过前 K 步，timer_steps=0 表示一直记。"""
    if step_index_1based <= start_step:
        return False
    if timer_steps <= 0:
        return True
    return step_index_1based <= (start_step + timer_steps)


class CommTimer:
    """用 CUDA/HIP 事件给 collective 计时；步头步尾圈住 train_batch()。"""

    # 这些函数名在 torch.distributed / deepspeed.comm / cdb.* 上都尽量包一下
    TARGET_FUNCS = [
        "all_reduce", "all_gather_into_tensor", "reduce_scatter_tensor",
        "broadcast", "all_to_all_single", "send", "recv",
        # 兼容旧名/别名
        "all_gather", "reduce_scatter", "isend", "irecv", "barrier",
    ]

    def __init__(self, enable: bool = True):
        self.enable = bool(enable and torch.cuda.is_available())
        self._installed = False
        self._pairs: List[Tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self._step_evt_start: Optional[torch.cuda.Event] = None

    # ---------- 公共 API ----------

    def install(self) -> None:
        """安装钩子：可以在分布式 init 之前或之后调用。"""
        if not self.enable or self._installed:
            return
        self._patch_all_namespaces()   # 先把能补的都补上
        self._attach_lazy_hooks()      # 拦截 DS 的 init_distributed/set_backend，等 cdb 建好再补一遍
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

        # 记录步尾
        step_end = torch.cuda.Event(enable_timing=True)
        step_end.record(torch.cuda.current_stream())
        torch.cuda.synchronize()

        # 整步 GPU 时长
        step_ms = float(self._step_evt_start.elapsed_time(step_end))

        # —— 把每个 collective 的起止事件，转换成相对步起点的 [t0, t1] 区间 —— 
        intervals = []
        for s, e in self._pairs:
            try:
                t0 = float(self._step_evt_start.elapsed_time(s))
                t1 = float(self._step_evt_start.elapsed_time(e))
                if t1 < t0:
                    t0, t1 = t1, t0
                # clamp 到 [0, step_ms]，并忽略极短/异常
                t0 = max(0.0, min(t0, step_ms))
                t1 = max(0.0, min(t1, step_ms))
                if t1 - t0 > 1e-3:
                    intervals.append((t0, t1))
            except Exception:
                # 个别事件测不出偏移就跳过
                pass

        # —— 合并区间（线段 union），得到通信覆盖时长 —— 
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

        # —— 兜底：如果一个都没并上，退回“求和再截断” —— 
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

    # ---------- 扁平化：下面全是内部小工具，互不嵌套 ----------

    def _wrap_collective_fn(self, fn: Callable) -> Callable:
        """模块函数/自由函数包装：调用前后在同一流打事件，不改函数签名，不同步。"""
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
        """类方法包装：对 cdb.__class__ 打补丁，避免实例绑定坑。"""
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
        """给某个命名空间（模块或对象）里的同名可调用成员打补丁。"""
        if space is None:
            return
        for n in names:
            if not hasattr(space, n):
                continue
            fn = getattr(space, n)
            if callable(fn) and not getattr(fn, "__wrapped_by_commtimer__", False):
                setattr(space, n, self._wrap_collective_fn(fn))

    def _patch_all_namespaces(self) -> None:
        """一次性对 torch.distributed 层、deepspeed.comm 层、已就绪的 cdb 进行补丁。"""
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

        # 2) deepspeed.comm 模块函数
        ds_comm = None
        try:
            import deepspeed
            ds_comm = deepspeed.comm
            self._wrap_module_space(ds_comm, self.TARGET_FUNCS)
        except Exception:
            ds_comm = None

        # 3) 如果 cdb 已就绪，再对其“类方法”打补丁（兜底）
        try:
            if ds_comm is not None and getattr(ds_comm, "cdb", None) is not None:
                cdb = ds_comm.cdb
                self._wrap_module_space(cdb, self.TARGET_FUNCS)  # 实例级兜底（多数实现无效，但无伤大雅）
                for n in self.TARGET_FUNCS:
                    self._wrap_class_method(cdb.__class__, n)
        except Exception:
            pass

    def _attach_lazy_hooks(self) -> None:
        """拦截 DS 的 init_distributed/set_backend，等 cdb 建好后再补一次。"""
        try:
            import deepspeed
            ds_comm = deepspeed.comm
        except Exception:
            return

        # 包一层：原函数返回后，再跑 _patch_all_namespaces
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
