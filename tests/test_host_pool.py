"""主机池的容量、对齐、回收身份和重置契约。"""
import ctypes
import mmap

import pytest

import tinyinfer.budget as budget_module
from tinyinfer.budget import HostBudget, HostPool


def pool(size=64):
    return HostPool.from_claimed(bytearray(size), size)


def offset(p, view):
    return ctypes.addressof(ctypes.c_char.from_buffer(view)) - p._base


def test_released_bytes_are_available_again():
    p = pool()
    a = p.alloc(32, align=1)
    b = p.alloc(32, align=1)
    assert (p.used, p.free) == (64, 0)
    p.release(offset(p, a), len(a))
    del a
    assert (p.used, p.free) == (32, 32)
    c = p.alloc(32, align=1)
    assert offset(p, c) == 0
    assert offset(p, b) == 32
    assert p.used + p.free == p.total


def test_alignment_is_preserved_when_reusing_a_hole():
    p = pool()
    a = p.alloc(3, align=1)
    b = p.alloc(13, align=1)
    c = p.alloc(8, align=1)
    p.release(offset(p, b), len(b))
    del b
    aligned = p.alloc(8, align=8)
    assert offset(p, aligned) == 8
    prefix = p.alloc(5, align=1)
    assert offset(p, prefix) == 3
    assert offset(p, c) == 16
    assert p.used == len(a) + len(c) + len(aligned) + len(prefix)


def test_padding_and_small_suffixes_are_not_lost():
    p = pool(17)
    a = p.alloc(9, align=1)
    b = p.alloc(1, align=16)
    c = p.alloc(7, align=1)
    assert [offset(p, x) for x in (a, b, c)] == [0, 16, 9]
    assert (p.used, p.free) == (17, 0)
    for view in (a, b, c):
        p.release(offset(p, view), len(view))
    assert p._free == [(0, 17)]
    assert p.free == 17


def test_failed_fit_does_not_change_accounting():
    p = pool()
    p.alloc(60, align=1)
    assert p.alloc(8, align=1) is None
    assert (p.used, p.free) == (60, 4)


def test_reset_clears_old_free_list_and_allocations():
    p = pool()
    a = p.alloc(16, align=1)
    p.alloc(16, align=1)
    p.release(offset(p, a), len(a))
    del a
    p.reset()
    whole = p.alloc(64, align=1)
    assert len(whole) == 64
    assert p.alloc(1, align=1) is None
    assert (p.used, p.free) == (64, 0)


@pytest.mark.parametrize("off,size", [(1, 16), (0, 15), (-1, 16), (0, 0)])
def test_invalid_release_cannot_create_free_bytes(off, size):
    p = pool()
    p.alloc(16, align=1)
    with pytest.raises(ValueError, match="unknown host allocation"):
        p.release(off, size)
    assert (p.used, p.free) == (16, 48)


def test_double_release_is_rejected():
    p = pool()
    p.alloc(16, align=1)
    p.release(0, 16)
    with pytest.raises(ValueError):
        p.release(0, 16)
    assert (p.used, p.free) == (0, 64)


def test_tensor_carries_its_reclaimable_identity():
    p = pool()
    t = p.tensor(12, align=1)
    t.fill_(7)
    assert t._tiny_pool_off == 0
    assert t._tiny_pool_len == 12
    assert t.data_ptr() == p._base
    off, size = t._tiny_pool_off, t._tiny_pool_len
    del t
    p.release(off, size)
    assert p.free == p.total


def test_default_alignment_uses_host_page_size():
    p = pool(2 * mmap.PAGESIZE)
    a = p.alloc(1)
    b = p.alloc(1)
    assert (p._base + offset(p, a)) % mmap.PAGESIZE == 0
    assert offset(p, b) - offset(p, a) == mmap.PAGESIZE


def test_declared_capacity_must_be_backed():
    with pytest.raises(ValueError, match="smaller"):
        HostPool.from_claimed(bytearray(4), 8)
    with pytest.raises(ValueError, match="writable"):
        HostPool.from_claimed(bytes(8), 8)

class FakeRuntime:
    def __init__(self, size, *, allocate_error=0, free_error=0):
        self.storage = ctypes.create_string_buffer(size)
        self.allocate_error = allocate_error
        self.free_error = free_error
        self.requests = []
        self.releases = []

    def cudaHostAlloc(self, pointer, size, flags):
        self.requests.append(size)
        if not self.allocate_error:
            ctypes.cast(pointer, ctypes.POINTER(ctypes.c_void_p))[0] = ctypes.addressof(self.storage)
        return self.allocate_error

    def cudaFreeHost(self, pointer):
        self.releases.append(pointer.value)
        return self.free_error


def budget(monkeypatch, runtime):
    import threading
    from types import SimpleNamespace

    b = HostBudget.__new__(HostBudget)
    b._lock = threading.Lock()
    b.limit = 128
    b.available = lambda: 128
    b._held = []
    b.granted = 0
    monkeypatch.setattr(budget_module, "_cudart", lambda: runtime)
    return b


def test_claim_is_exact_even_below_old_minimum(monkeypatch):
    runtime = FakeRuntime(32)
    b = budget(monkeypatch, runtime)
    assert b.claim(32) == 32
    assert runtime.requests == [32]
    assert b.granted == b._claim_bytes == 32
    assert b._claim_pinned
    assert b.close() == 32
    assert b.granted == 0


def test_failed_claim_does_not_retry_smaller_or_use_pageable_memory(monkeypatch):
    runtime = FakeRuntime(32, allocate_error=2)
    b = budget(monkeypatch, runtime)
    with pytest.raises(MemoryError, match="CUDA error=2"):
        b.claim(32)
    assert runtime.requests == [32]
    assert b.granted == 0
    assert not getattr(b, "_held", [])


def test_claim_over_budget_fails_before_allocation(monkeypatch):
    runtime = FakeRuntime(256)
    b = budget(monkeypatch, runtime)
    with pytest.raises(MemoryError, match="exceeds budget"):
        b.claim(256)
    assert not runtime.requests
    assert b.granted == 0


def test_missing_cuda_runtime_is_not_a_pageable_fallback(monkeypatch):
    b = budget(monkeypatch, None)
    with pytest.raises(RuntimeError, match="CUDA runtime"):
        b.claim(32)
    assert b.granted == 0


def test_failed_release_keeps_claim_and_can_be_retried(monkeypatch):
    runtime = FakeRuntime(32, free_error=2)
    b = budget(monkeypatch, runtime)
    b.claim(32)
    with pytest.raises(RuntimeError, match="release incomplete"):
        b.close()
    assert b.granted == b._claim_bytes == 32
    assert len(b._held) == 1
    runtime.free_error = 0
    assert b.close() == 32
    assert b.granted == 0
    assert b._claim_buf is None


def test_alignment_uses_actual_address_for_sliced_backing():
    backing = memoryview(bytearray(128))[1:]
    p = HostPool.from_claimed(backing, len(backing))
    view = p.alloc(8, align=16)
    assert (p._base + offset(p, view)) % 16 == 0
    assert p.used == 8
    assert p.free == p.total - 8


def test_dense_model_does_not_enter_expert_pool_creation():
    import ast
    import inspect
    import textwrap
    from types import SimpleNamespace
    from tinyinfer.engine import Engine
    from tinyinfer.tiers import OptLevel

    source = ast.parse(textwrap.dedent(inspect.getsource(Engine.__init__)))
    branch = next(node for node in source.body[0].body
                  if isinstance(node, ast.If) and "expert_store is None" in ast.unparse(node.test))
    predicate = compile(ast.Expression(branch.test), "expert-pool-condition", "eval")
    for count, expected in ((0, False), (8, True)):
        engine = SimpleNamespace(hw_plan=object(), cfg=SimpleNamespace(n_routed_experts=count),
                                 tier=SimpleNamespace(level=OptLevel.AGGRESSIVE))
        assert eval(predicate, {"self": engine, "expert_store": None, "OptLevel": OptLevel}) is expected


def test_budget_ignores_percentage_and_reserve_overrides(monkeypatch):
    import psutil
    from types import SimpleNamespace

    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(total=1024, available=768))
    monkeypatch.setenv("TI_RAM_FRACTION", "99")
    monkeypatch.setenv("TI_RAM_RESERVE_GB", "999")
    b = HostBudget()
    assert b.total == 1024
    assert b.limit == 768
    assert b.available() == 768
    assert not hasattr(b, "reserve")


def test_failed_capacity_read_is_not_replaced_by_old_snapshot(monkeypatch):
    import psutil

    def fail():
        raise OSError("physical memory query failed")
    monkeypatch.setattr(psutil, "virtual_memory", fail)
    with pytest.raises(OSError, match="query failed"):
        HostBudget()
    with pytest.raises(OSError, match="query failed"):
        budget_module.physical_available()
    assert not hasattr(budget_module, "available_commit")


def test_zero_available_memory_remains_zero(monkeypatch):
    import psutil
    from types import SimpleNamespace

    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(total=1024, available=0))
    b = HostBudget()
    assert b.room() == 0
    with pytest.raises(MemoryError):
        b.grant(1)


def test_grants_are_exact_instead_of_shrinking(monkeypatch):
    b = budget(monkeypatch, FakeRuntime(32))
    assert b.grant(96) == 96
    with pytest.raises(MemoryError, match="grant exceeds"):
        b.grant(64)
    assert b.granted == 96
    b.release(96)
    assert b.granted == 0


def test_owned_memory_cannot_be_freed_by_releasing_a_logical_grant(monkeypatch):
    b = budget(monkeypatch, FakeRuntime(32))
    b.claim(32)
    with pytest.raises(ValueError, match="outstanding host grant"):
        b.release(32)
    assert b.granted == 32
    assert b.close() == 32


def test_new_allocation_uses_the_planned_capacity_after_previous_pool_is_freed(monkeypatch):
    runtime = FakeRuntime(256)
    b = budget(monkeypatch, runtime)
    b.claim(32)
    b.close()
    b.set_limit(256)
    assert b.claim(192) == 192
    assert b.limit == 256
    assert b.close() == 192


def test_limit_cannot_be_rewritten_while_memory_is_owned(monkeypatch):
    b = budget(monkeypatch, FakeRuntime(256))
    b.claim(32)
    with pytest.raises(RuntimeError, match="already owns memory"):
        b.set_limit(512)
    assert b.limit == min(b.limit, 256)
    b.close()


def test_real_claimed_pool_can_feed_async_cuda_copy():
    import torch
    from tinyinfer.loading import _async_h2d

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    b = HostBudget()
    size = mmap.PAGESIZE
    assert b.claim(size) == size
    host = torch.frombuffer(b._claim_buf, dtype=torch.uint8).fill_(7)
    device = torch.empty(size, dtype=torch.uint8, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    assert _async_h2d(device, host, size, stream=stream)
    stream.synchronize()
    assert torch.equal(device.cpu(), host)
    del host
    assert b.close() == size
    assert b.granted == 0


@pytest.mark.parametrize("base_offset", [0, 1])
def test_repeated_fragmented_allocations_preserve_live_views(base_offset):
    import random

    rng = random.Random(1729)
    backing = memoryview(bytearray(129))[base_offset:base_offset + 128]
    p = HostPool.from_claimed(backing, 128)
    live = []
    for step in range(300):
        if live and rng.randrange(3) == 0:
            index = rng.randrange(len(live))
            view, value = live.pop(index)
            p.release(offset(p, view), len(view))
            del view
        else:
            size = rng.randrange(1, 18)
            alignment = 1 << rng.randrange(4)
            view = p.alloc(size, align=alignment)
            if view is not None:
                assert (p._base + offset(p, view)) % alignment == 0
                value = step % 256
                view[:] = bytes([value]) * size
                live.append((view, value))
        assert p.used == sum(len(view) for view, _ in live)
        assert p.used + p.free == p.total
        spans = sorted((offset(p, view), offset(p, view) + len(view)) for view, _ in live)
        assert all(left[1] <= right[0] for left, right in zip(spans, spans[1:]))
        assert all(bytes(view) == bytes([value]) * len(view) for view, value in live)
