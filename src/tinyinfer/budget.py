"""主机内存分配与记账。

HostPool 管理已申请存储中的切片、对齐和回收，不决定容量。
HostBudget 执行容量申请；锁页申请失败时不缩容或改用可换出存储。
"""
from __future__ import annotations

import threading

GB = 1 << 30


def _memory_snapshot() -> tuple[int, int]:
    """读取操作系统的物理内存总量与可用量；读取失败必须上抛。"""
    import psutil

    memory = psutil.virtual_memory()
    total, available = int(memory.total), int(memory.available)
    if total <= 0 or not 0 <= available <= total:
        raise RuntimeError("invalid physical memory snapshot")
    return total, available


def physical_available() -> int:
    return _memory_snapshot()[1]


_CUDART = None


def _cudart():
    """Locate the CUDA runtime that ships with torch (for cudaHostAlloc)."""
    global _CUDART
    if _CUDART is not False and _CUDART is not None:
        return _CUDART
    if _CUDART is False:
        return None
    try:
        import ctypes
        import glob
        import os

        import torch

        lib_dir = os.path.join(os.path.dirname(torch.__file__), "lib")
        # Windows 的 cudart 是 cudart64_XX.dll；Linux 是 libcudart.so.XX。
        # 以前只 glob 了 .dll，于是**在 Linux（L20 部署机）上永远找不到 cudart**，
        # async 专家流水线直接降级成逐层同步路径 —— 实测 L20 上刷了 20 行
        # "[async] expert pipeline unavailable (cudart not found)"，
        # 那正是「GPU 主动下单、CPU 立刻推送」这条设计的载体。
        _cands = (glob.glob(os.path.join(lib_dir, "cudart*.dll"))
                  + glob.glob(os.path.join(lib_dir, "libcudart*.so*")))
        # torch 2.15 的 Linux wheel **不带** libcudart（它静态链接了 CUDA 运行时），
        # 真正的 so 在 nvidia/cu13/lib 或系统 CUDA 目录。只 glob torch/lib 会
        # 在 L20 上永远返回 False。这里再按顺序试这些位置。
        try:
            import sysconfig as _sc

            _sp = _sc.get_paths().get("purelib") or ""
        except Exception:
            _sp = ""
        for _pat in ("nvidia/cu13/lib/libcudart.so*", "nvidia/cuda_runtime/lib/libcudart.so*"):
            if _sp:
                _cands += glob.glob(os.path.join(_sp, _pat))
        _cands += ["libcudart.so", "libcudart.so.13", "libcudart.so.12"]
        for path in _cands:
            try:
                lib = ctypes.CDLL(path)
                lib.cudaHostAlloc.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                                              ctypes.c_size_t, ctypes.c_uint]
                lib.cudaHostAlloc.restype = ctypes.c_int
                lib.cudaFreeHost.argtypes = [ctypes.c_void_p]
                lib.cudaFreeHost.restype = ctypes.c_int
                _CUDART = lib
                return lib
            except Exception:
                continue
    except Exception:
        pass
    _CUDART = False
    return None


_PINNED_LIB = None
_PINNED_FREE: list = []


class PinnedBuffer:
    """cudaHostAlloc(Mapped) 的 mmap 兼容包装：主机可读写，GPU 有 device 指针。

    kernel 直接拿 device 指针读这块内存（实测 24.5 GB/s，满速 PCIe），所以
    decode 期间 CPU 不需要参与任何搬运 —— 请求由 GPU 发起。

    L20 的 cudaDevAttrUnifiedAddressing = 0，pageable 内存（mmap）没有可用的
    device 指针，因此必须 pin。pin 的实测速度 1.4 GB/s（8GB 4.3s / 32GB 16.3s /
    128GB 89.9s），275GB 约 3.3 分钟，是一次性启动成本。
    """
    __slots__ = ("_cpu", "_dev", "_n")

    def __init__(self, cpu_ptr: int, dev_ptr: int, n: int):
        self._cpu = int(cpu_ptr)
        self._dev = int(dev_ptr)
        self._n = int(n)

    def __len__(self) -> int:
        return self._n

    def addressof(self) -> int:
        """GPU 侧指针（kernel 用这个）。"""
        return self._dev

    def cpu_address(self) -> int:
        return self._cpu

    def __getitem__(self, k):
        import ctypes as _c
        if isinstance(k, slice):
            start, stop, step = k.indices(self._n)
            if step != 1:
                raise ValueError("PinnedBuffer slice step must be 1")
            return _c.string_at(self._cpu + start, stop - start)
        return _c.c_ubyte.from_address(self._cpu + k).value

    def __setitem__(self, k, v) -> None:
        import ctypes as _c
        if isinstance(k, slice):
            start, stop, step = k.indices(self._n)
            if step != 1:
                raise ValueError("PinnedBuffer slice step must be 1")
            _c.memmove(self._cpu + start, bytes(v), min(len(v), stop - start))
        else:
            _c.c_ubyte.from_address(self._cpu + k).value = int(v) & 0xFF

    def __buffer__(self, flags: int):
        import ctypes as _c
        return memoryview((_c.c_ubyte * self._n).from_address(self._cpu))

    def __release_buffer__(self, view) -> None:
        pass

    def close(self) -> None:
        pass


def _pinned_pool(nbytes: int):
    """mmap 出内存后分块并行 cudaHostRegister，取得 GPU device 指针。

    cudaHostAlloc 内部 pin 是单线程的：实测 8GB 4.3s、32GB 16.3s、128GB 89.9s
    （1.4-1.9 GB/s），275GB 要三分多钟，每次启动都要等。

    但 pin 是内核逐页操作，天然可分块并行。cudaHostRegister 分块从 4 个线程
    起就是 19.1 GB/s（32GB 1.7s，1 线程 16.5s）—— 快 10 倍，275GB 约 14 秒。
    """
    import ctypes as _c
    import mmap
    import os as _os
    global _PINNED_LIB
    if _PINNED_LIB is None:
        try:
            lib = _c.CDLL("libcudart.so")
            lib.cudaHostRegister.argtypes = [_c.c_void_p, _c.c_size_t, _c.c_uint]
            lib.cudaHostRegister.restype = _c.c_int
            lib.cudaHostUnregister.argtypes = [_c.c_void_p]
            lib.cudaHostUnregister.restype = _c.c_int
            lib.cudaHostGetDevicePointer.argtypes = [_c.POINTER(_c.c_void_p),
                                                    _c.c_void_p, _c.c_uint]
            lib.cudaHostGetDevicePointer.restype = _c.c_int
            lib.cudaSetDevice.restype = _c.c_int
            _PINNED_LIB = lib
        except Exception:
            _PINNED_LIB = False
    lib = _PINNED_LIB
    if not lib:
        return None

    n = int(nbytes)
    buf = mmap.mmap(-1, n)
    addr = _c.addressof(_c.c_char.from_buffer(buf))
    lib.cudaSetDevice(0)

    nthreads = max(1, min(8, (_os.cpu_count() or 4)))
    chunk = (n + nthreads - 1) // nthreads

    def _reg(off: int, size: int) -> int:
        return lib.cudaHostRegister(_c.c_void_p(addr + off), _c.c_size_t(size), 2)

    import threading
    results = [None] * nthreads
    def _worker(i: int) -> None:
        off = i * chunk
        size = min(chunk, n - off)
        if size > 0:
            results[i] = _reg(off, size)

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(nthreads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if any(r != 0 for r in results):
        raise MemoryError(
            f"cudaHostRegister failed for the {n / (1 << 30):.1f}GB expert pool: "
            f"{results}. Raise the memlock limit (ulimit -l / RLIMIT_MEMLOCK).")

    dp = _c.c_void_p()
    rc = lib.cudaHostGetDevicePointer(_c.byref(dp), _c.c_void_p(addr), 0)
    if rc != 0 or not dp.value:
        lib.cudaHostUnregister(_c.c_void_p(addr))
        raise MemoryError(
            f"cudaHostGetDevicePointer on the {n / (1 << 30):.1f}GB expert pool "
            f"failed (rc={rc}); the pool would not be GPU addressable.")

    _PINNED_FREE.append((buf, addr))
    return PinnedBuffer(addr, dp.value, n)


class HostPool:
    """已分配主机内存的子分配器；配额由规划器决定，余量由活跃切片决定。"""

    def __init__(self, total: int):
        self.total = int(total)
        if self.total < 0:
            raise ValueError("host pool capacity cannot be negative")
        self._view = None
        self._owner = None
        self._base = None
        self._lock = threading.Lock()
        self._free = [(0, self.total)] if self.total else []
        self._allocations: dict[int, int] = {}
        self._used = 0

    @classmethod
    def from_claimed(cls, buffer, nbytes: int) -> "HostPool":
        import ctypes

        pool = cls(nbytes)
        view = memoryview(buffer).cast("B")
        if view.readonly:
            raise ValueError("host pool backing must be writable")
        if view.nbytes < pool.total:
            raise ValueError("host pool backing is smaller than its declared capacity")
        pool._view = view[:pool.total]
        pool._owner = buffer
        if pool.total:
            pool._base = ctypes.addressof(ctypes.c_char.from_buffer(pool._view))
        return pool

    def slice_at(self, offset: int, n: int):
        """借用已持有内存的视图，不创建新的分配记录。"""
        if self._view is None or offset < 0 or n <= 0 or offset + n > self.total:
            return None
        return self._view[offset:offset + n]

    @property
    def used(self) -> int:
        with self._lock:
            return self._used

    @property
    def free(self) -> int:
        with self._lock:
            return self.total - self._used

    def release(self, offset: int, n: int) -> None:
        """归还完整分配；调用方必须先解除所有持有该切片的引用。"""
        off, size = int(offset), int(n)
        with self._lock:
            if self._allocations.get(off) != size or size <= 0:
                raise ValueError(f"unknown host allocation: offset={off}, bytes={size}")
            del self._allocations[off]
            self._used -= size
            self._free.append((off, size))
            self._free.sort()
            merged = []
            for start, length in self._free:
                if merged and merged[-1][0] + merged[-1][1] == start:
                    previous, previous_size = merged[-1]
                    merged[-1] = (previous, previous_size + length)
                else:
                    merged.append((start, length))
            self._free = merged

    def alloc(self, n: int, align: int | None = None) -> memoryview | None:
        """按空闲区首次适配；对齐产生的前后碎片仍归池所有。"""
        import mmap

        size = int(n)
        alignment = mmap.PAGESIZE if align is None else int(align)
        if size <= 0 or alignment <= 0:
            raise ValueError("allocation size and alignment must be positive")
        if self._view is None:
            return None
        with self._lock:
            for idx, (off, length) in enumerate(self._free):
                start = (self._base + off + alignment - 1) // alignment * alignment - self._base
                end = start + size
                if end > off + length:
                    continue
                fragments = []
                if start > off:
                    fragments.append((off, start - off))
                if end < off + length:
                    fragments.append((end, off + length - end))
                self._free[idx:idx + 1] = fragments
                self._allocations[start] = size
                self._used += size
                return self._view[start:end]
        return None

    def tensor(self, n: int, align: int | None = None):
        """创建带回收身份的 uint8 视图；不另分配主机存储。"""
        mv = self.alloc(n, align=align)
        if mv is None:
            return None
        import ctypes
        import torch

        offset = ctypes.addressof(ctypes.c_char.from_buffer(mv)) - self._base
        try:
            tensor = torch.frombuffer(mv, dtype=torch.uint8)
        except Exception:
            self.release(offset, n)
            raise
        tensor._tiny_pool_mv = mv
        tensor._tiny_pool_off = offset
        tensor._tiny_pool_len = int(n)
        return tensor

    def reset(self) -> None:
        """调用方解除全部视图后，重置整个池的分配记录。"""
        with self._lock:
            self._free = [(0, self.total)] if self.total else []
            self._allocations.clear()
            self._used = 0


class HostBudget:
    def __init__(self):
        self._lock = threading.Lock()
        self.total, self.limit = _memory_snapshot()
        self.granted = 0
        self._held = []

    def available(self) -> int:
        return physical_available()

    def set_limit(self, nbytes: int) -> None:
        """把额度固定为规划时的那一份物理余量，此后不再重新采样。"""
        nbytes = int(nbytes)
        if nbytes <= 0:
            raise ValueError("host budget limit must be positive")
        with self._lock:
            if self.granted or self._held:
                raise RuntimeError("cannot change a host budget that already owns memory")
            self.limit = nbytes

    def claim(self, nbytes: int) -> int:
        """精确申请真实锁页内存；和只做额度登记的 grant 不同。"""
        nbytes = int(nbytes)
        if nbytes <= 0:
            raise ValueError("host pool claim must be positive")
        with self._lock:
            if nbytes > self.room():
                raise MemoryError(f"host claim exceeds budget: requested={nbytes}")
            got = self._commit(nbytes)
            if got != nbytes:
                raise RuntimeError(f"host allocator returned {got} bytes for an exact {nbytes}-byte claim")
            self.granted += nbytes
            return nbytes

    def _commit(self, nbytes: int) -> int:
        """精确申请所需字节；锁页失败不缩容，也不改用可换出的内存。"""
        import ctypes

        runtime = _cudart()
        if runtime is None:
            raise RuntimeError("CUDA runtime is required for the pinned host pool")
        ptr = ctypes.c_void_p()
        rc = runtime.cudaHostAlloc(ctypes.byref(ptr), nbytes, 0)
        if rc != 0 or not ptr.value:
            raise MemoryError(f"cannot allocate pinned host pool: bytes={nbytes}, CUDA error={rc}")
        try:
            view = (ctypes.c_char * nbytes).from_address(ptr.value)
            if getattr(self, "_held", None) is None:
                self._held = []
            self._held.append(("pinned", runtime, ptr.value, nbytes, view))
        except Exception:
            runtime.cudaFreeHost(ptr)
            raise
        self._claim_buf = view
        self._claim_bytes = nbytes
        self._claim_pinned = True
        return nbytes

    def close(self) -> int:
        """释放真实持有的分配；失败的分配仍保留身份和记账，允许稍后重试。"""
        import ctypes

        freed = 0
        errors = []
        with self._lock:
            remaining = []
            for item in reversed(getattr(self, "_held", [])):
                try:
                    if item[0] == "pinned":
                        _, runtime, ptr, nbytes, view = item
                        rc = runtime.cudaFreeHost(ctypes.c_void_p(ptr))
                        if rc != 0:
                            raise RuntimeError(f"cudaFreeHost failed: CUDA error={rc}")
                    else:
                        raise RuntimeError("unknown host allocation kind")
                    freed += nbytes
                except Exception as exc:
                    remaining.append(item)
                    errors.append(str(exc))
            self._held = list(reversed(remaining))
            self.granted -= freed
            if self._held:
                last = self._held[-1]
                self._claim_pinned = True
                self._claim_buf = last[4]
                self._claim_bytes = last[3]
            else:
                self._claim_buf = None
                self._claim_bytes = 0
                self._claim_pinned = False
        if errors:
            raise RuntimeError("host pool release incomplete: " + "; ".join(errors))
        return freed

    def room(self) -> int:
        """未占用的额度；额度是规划时固定下来的那份余量。"""
        return max(0, self.limit - self.granted)

    def grant(self, want: int) -> int:
        """登记精确的外部分配额度；不足时不返回一个缩水的数字。"""
        want = int(want)
        if want < 0:
            raise ValueError("host grant cannot be negative")
        with self._lock:
            if want > self.room():
                raise MemoryError(f"host grant exceeds budget: requested={want}")
            self.granted += want
            return want

    def release(self, nbytes: int) -> None:
        """归还外部分配额度；真实持有的锁页块必须经 close 释放。"""
        nbytes = int(nbytes)
        with self._lock:
            owned = sum(item[3] for item in self._held)
            if nbytes < 0 or nbytes > self.granted - owned:
                raise ValueError("release does not match an outstanding host grant")
            self.granted -= nbytes

    def stats(self) -> dict:
        return {
            "total_gb": self.total / GB,
            "avail_gb": self.available() / GB,
            "granted_gb": self.granted / GB,
        }


_BUDGET: HostBudget | None = None


def budget() -> HostBudget:
    """Process-wide budget (created on first use)."""
    global _BUDGET
    if _BUDGET is None:
        _BUDGET = HostBudget()
    return _BUDGET
