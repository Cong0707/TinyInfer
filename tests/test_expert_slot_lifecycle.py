"""Reader leases must outlive all native CUDA consumers of a slot."""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock
import threading

import pytest
import torch

from tinyinfer.deepseek.moe import _expert_batches
from tinyinfer.expert_tiers import ExpertTierManager
from tinyinfer.loading import ExpertStore, VramSlotCache


def source(layer, experts):
    return {expert: bytearray([expert % 256]) * 32 for expert in experts}


def pool(capacity=3, device="cpu"):
    value = VramSlotCache(capacity, 32, device=device, slot_arena=True)
    value._ensure()
    return value


def manager(cache):
    store = SimpleNamespace(slots=cache, ram=None, index=None, reader=None)
    value = ExpertTierManager(store, n_layers=4, n_experts=64, warm=False)
    value._provide = source
    return value


def test_complete_request_preserves_oldest_hit_even_when_miss_is_first():
    cache = pool()
    cache.ensure(0, [1, 2, 3], source)
    old_slot = cache._by_key[(0, 1)]
    cache._last_used[old_slot] = -1
    mgr = manager(cache)
    mgr.need(0, [4, 1])
    assert cache._by_key[(0, 1)] == old_slot
    assert old_slot in cache._urgent
    mgr.commit(0)
    lease = mgr.ready(0, [4, 1])
    assert lease == [4, 1]
    for expert in lease:
        assert bool((cache.buffer(cache._by_key[(0, expert)]) == expert).all())
    mgr.release(0, lease)
    assert not any(cache._refs)
    assert not cache._urgent


def test_request_is_held_between_need_commit_and_ready():
    cache = pool(2)
    cache.ensure(0, [1, 2], source)
    mgr = manager(cache)
    assert mgr.need(0, [1, 2]) == 0
    assert not cache.preload(1, 9, source(1, [9])[9])
    lease = mgr.ready(0, [1, 2])
    mgr.layer_begin()
    assert not cache.preload(1, 9, source(1, [9])[9])
    mgr.release(0, lease)
    assert cache.preload(1, 9, source(1, [9])[9])


def test_over_capacity_request_fails_without_reading_or_evicting():
    cache = pool(2)
    cache.ensure(0, [1, 2], source)
    before = dict(cache._by_key)
    loader = Mock(side_effect=AssertionError("must validate capacity before reading"))
    with pytest.raises(RuntimeError, match="exceeds available slots"):
        cache.stage(0, [1, 2, 3], loader)
    assert cache._by_key == before
    assert not cache._urgent
    loader.assert_not_called()


def test_bad_source_rolls_back_protection_and_all_reservations():
    cache = pool()
    cache.ensure(0, [1], source)
    with pytest.raises(ValueError, match="planned slot size"):
        cache.stage(0, [1, 2, 3], lambda layer, wanted: {2: bytearray(32), 3: bytearray(31)})
    assert set(cache._by_key) == {(0, 1)}
    assert not cache._urgent
    assert not cache._stage_bufs


def test_batch_acquire_protects_late_hits_before_reserving_misses():
    cache = pool()
    cache.ensure(0, [1, 2, 3], source)
    old_slot = cache._by_key[(0, 1)]
    cache._last_used[old_slot] = -1
    acquired = cache.acquire_many(0, [4, 1])
    assert acquired[1] == (old_slot, True)
    index, ready = acquired[0]
    assert not ready
    cache.publish(index, cache.copy_in(index, source(0, [4])[4]))
    cache.release_many(0, [4, 1])
    assert not any(cache._refs)


def test_ready_rejects_unpublished_transfer_atomically():
    cache = pool(2)
    cache.ensure(0, [1], source)
    cache.stage(0, [1, 2], source)
    with pytest.raises(RuntimeError, match="not been published"):
        cache.acquire_ready(0, [1, 2])
    assert not any(cache._refs)
    cache.cancel_request(0, [1, 2])
    assert set(cache._by_key) == {(0, 1)}
    assert not cache._urgent


def test_release_without_lease_does_not_change_storage():
    cache = pool(1)
    cache.ensure(0, [9], source)
    before = dict(cache._by_key)
    with pytest.raises(RuntimeError, match="without a read lease"):
        cache.release(0, 9)
    assert cache._by_key == before


def test_failed_dma_does_not_publish_or_change_copy_implementation(monkeypatch):
    cache = pool(2)
    cache.stage(0, [1, 2], source)
    copy = Mock(side_effect=RuntimeError("DMA rejected"))
    monkeypatch.setattr(cache, "_copy_blocking", copy)
    with pytest.raises(RuntimeError, match="DMA rejected"):
        cache.commit(0)
    assert copy.call_count == 1
    assert not cache._by_key
    assert not cache._stage_bufs
    assert all(state == cache.FREE for state in cache._state)


def test_arena_pointer_arithmetic_covers_more_than_old_chunk_size():
    cache = pool(67)
    base = cache._arena[0].data_ptr()
    assert len(cache._arena) == 1
    assert len(cache._buffers) == cache.slots
    assert all(buf.data_ptr() == base + index * cache.blob_bytes
               for index, buf in enumerate(cache._buffers))


def test_arena_allocation_failure_preserves_planned_capacity(monkeypatch):
    cache = VramSlotCache(67, 32, slot_arena=True, device="cpu")
    allocation = Mock(side_effect=RuntimeError("out of memory"))
    monkeypatch.setattr(torch, "empty", allocation)
    with pytest.raises(RuntimeError, match="out of memory"):
        cache._ensure()
    assert allocation.call_count == 1
    assert cache.slots == 67
    assert not cache._buffers


def test_explicit_drop_preserves_capacity_and_requires_no_owners():
    cache = pool(3)
    store = ExpertStore.__new__(ExpertStore)
    store.slots = cache
    cache.ensure(0, [1, 2], source)
    cache.acquire_ready(0, [1, 2])
    with pytest.raises(RuntimeError, match="active readers or producers"):
        store.drop_slots()
    cache.release_many(0, [1, 2])
    store.drop_slots()
    assert cache.slots == 3
    assert not cache._buffers
    assert cache._slot_tab is None
    store.resume_slots()
    cache.ensure(0, [3], source)
    assert bool((cache.buffer(cache._by_key[(0, 3)]) == 3).all())


def test_drop_rejects_unsubmitted_producer():
    store = ExpertStore.__new__(ExpertStore)
    store.slots = pool(1)
    store.slots.stage(0, [1], source)
    with pytest.raises(RuntimeError, match="active readers or producers"):
        store.drop_slots()
    assert store.slots._stage_bufs


def test_fetch_async_returns_future_and_executes_fetch():
    store = ExpertStore.__new__(ExpertStore)
    store.fetch = Mock(return_value=["record"])
    with ThreadPoolExecutor(max_workers=1) as executor:
        store._fetch_pool = executor
        assert store.fetch_async(2, [7], cache=False).result(timeout=5) == ["record"]
    store.fetch.assert_called_once_with(2, [7], cache=False)


def test_store_context_manager_releases_slot_and_ram_lease():
    store = ExpertStore.__new__(ExpertStore)
    store.slots = pool(1)
    store.slots.ensure(0, [7], source)
    store.slots.acquire_ready(0, [7])
    store._lock = threading.Lock()
    store._open = defaultdict(int, {(0, 7): 1})
    store.ram = SimpleNamespace(lease=Mock())
    record = SimpleNamespace(key=(0, 7), event=None)
    store.fetch = Mock(return_value=[record])
    with store.experts(0, [7]) as records:
        assert records == [record]
    assert store._open[(0, 7)] == 0
    assert store.slots._refs == [0]
    store.ram.lease.assert_called_once_with((0, 7), -1)


@pytest.mark.parametrize(("routes", "capacity", "expected"), [
    ([[1, 2], [2, 3], [7, 8]], 3, [(0, 2), (2, 3)]),
    ([[1, 2]] * 9, 2, [(0, 9)]),
    ([[1], [2], [3]], 1, [(0, 1), (1, 2), (2, 3)]),
    ([[1, 2], [3, 4]], 4, [(0, 2)]),
])
def test_batches_cover_actual_route_union(routes, capacity, expected):
    assert _expert_batches(routes, capacity) == expected
    assert [row for first, last in expected for row in routes[first:last]] == routes
    for first, last in expected:
        assert len({e for row in routes[first:last] for e in row}) <= capacity


def test_one_row_does_not_silently_drop_experts():
    with pytest.raises(RuntimeError, match="exceeds planned expert capacity"):
        _expert_batches([[1, 2, 3]], 2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("reader_count", [1, 2])
def test_overwrite_waits_for_every_consumer_stream(reader_count):
    cache = pool(1, "cuda")
    first = torch.full((32,), 7, dtype=torch.uint8, pin_memory=True)
    second = torch.full((32,), 11, dtype=torch.uint8, pin_memory=True)
    index, ready = cache.acquire(0, 7)
    assert not ready
    cache.publish(index, cache.copy_in(index, first))
    cache.release(0, 7)
    outputs = []
    streams = [torch.cuda.Stream() for _ in range(reader_count)]
    for stream in streams:
        with torch.cuda.stream(stream):
            cache.wait_ready(cache.acquire_ready(0, [7]))
            torch.cuda._sleep(5_000_000)
            outputs.append(cache.buffer(index).clone())
            cache.release(0, 7)
    assert len(cache._consumer_events[index]) >= reader_count
    index, ready = cache.acquire(0, 11)
    assert not ready
    cache.publish(index, cache.copy_in(index, second))
    cache.wait_ready([index])
    current = cache.buffer(index).clone()
    cache.release(0, 11)
    torch.cuda.synchronize()
    assert all(bool((output == 7).all()) for output in outputs)
    assert bool((current == 11).all())
    assert not any(cache._refs)
    cache.synchronize()
    assert not cache._dma_sources


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_graph_consumer_event_protects_replayed_reads():
    cache = pool(1, "cuda")
    cache.ensure(0, [7], source)
    index = cache.acquire_ready(0, [7])[0]
    cache.wait_ready([index])
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    output = torch.empty_like(cache.buffer(index))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output.copy_(cache.buffer(index))
    with torch.cuda.stream(stream):
        torch.cuda._sleep(5_000_000)
        graph.replay()
        cache.release(0, 7)
    cache.ensure(0, [11], source)
    torch.cuda.synchronize()
    assert bool((output == 7).all())
    assert bool((cache.buffer(index) == 11).all())
    cache.synchronize()


class PendingEvent:
    def __init__(self):
        self.done = False

    def query(self):
        return self.done

    def synchronize(self):
        self.done = True


def source_manager(cache):
    mgr = manager(cache)
    refs = defaultdict(int)

    def lease(key, delta):
        refs[key] += delta
        assert refs[key] >= 0

    def provide(layer, wanted):
        assert all(refs[(layer, expert)] > 0 for expert in wanted)
        return source(layer, wanted)

    mgr.ram = SimpleNamespace(lease=lease)
    mgr._provide = provide
    return mgr, refs


def test_host_source_lease_remains_until_dma_completion(monkeypatch):
    cache = pool(1)
    mgr, refs = source_manager(cache)
    event = PendingEvent()
    copy = cache._copy_blocking

    def submit(index, data):
        copy(index, data)
        cache._copy_events[index] = event
        return event

    monkeypatch.setattr(cache, "_copy_blocking", submit)
    mgr.need(0, [7])
    assert refs[(0, 7)] == 1
    mgr.commit(0)
    mgr.drain_sources()
    assert refs[(0, 7)] == 1
    event.done = True
    mgr.drain_sources()
    assert refs[(0, 7)] == 0


def test_partial_commit_failure_retains_only_submitted_sources(monkeypatch):
    cache = pool(2)
    mgr, refs = source_manager(cache)
    event = PendingEvent()
    count = 0

    def submit(index, data):
        nonlocal count
        count += 1
        if count == 2:
            raise RuntimeError("second DMA rejected")
        cache._copy_events[index] = event
        return event

    monkeypatch.setattr(cache, "_copy_blocking", submit)
    mgr.need(0, [7, 9])
    with pytest.raises(RuntimeError, match="second DMA rejected"):
        mgr.commit(0)
    assert refs[(0, 7)] == 1
    assert refs[(0, 9)] == 0
    mgr.close()
    assert refs[(0, 7)] == 0
    assert event.done


def test_source_read_failure_releases_host_leases():
    cache = pool(2)
    mgr, refs = source_manager(cache)
    mgr._provide = Mock(side_effect=OSError("disk read failed"))
    with pytest.raises(OSError, match="disk read failed"):
        mgr.need(0, [7, 9])
    assert all(value == 0 for value in refs.values())
    assert not cache._stage_bufs
    assert not cache._urgent


def test_close_cancels_unsubmitted_sources():
    cache = pool(2)
    mgr, refs = source_manager(cache)
    mgr.need(0, [7, 9])
    mgr.close()
    assert all(value == 0 for value in refs.values())
    assert not cache._stage_bufs
    assert not cache._by_key


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("kernel_failure", [False, True])
def test_moe_dispatch_reads_actual_routes_and_releases_every_batch(monkeypatch, kernel_failure):
    from tinyinfer.deepseek import moe as module
    from tinyinfer.kernels import native, quant

    cache = pool(3, "cuda")
    mgr = manager(cache)
    layer = module.MoE.__new__(module.MoE)
    layer.cfg = SimpleNamespace(dim=256, moe_inter_dim=256, n_routed_experts=64)
    layer.layer_id = 0
    layer._expert_store = SimpleNamespace(slots=cache)
    layer._mgr = mgr
    layer._route_counts = {}
    layer._last_ids = [63]
    layer.swiglu_limit = 1.0
    layer._ts_offsets = lambda _: (0, 0, 0, 0, 0, 0)
    ids = torch.tensor([[1, 2], [2, 3], [5, 6]], device="cuda")
    weights = torch.tensor([[0.25, 0.75]] * 3, device="cuda")
    layer.gate = SimpleNamespace(forward=lambda *_: (weights, ids))
    layer.shared = SimpleNamespace(forward=lambda x: torch.full_like(x, 2))
    pending = {}

    def act(*args, **kwargs):
        if kernel_failure:
            raise RuntimeError("expert launch failed")
        arena, slot_ids, route_weights = args[2:5]
        pending["value"] = arena[slot_ids.long(), 0].float() * route_weights

    def down(*args, **kwargs):
        output = args[4]
        rows = output.shape[0]
        value = pending["value"].view(rows, -1).sum(dim=1, keepdim=True)
        output.copy_(value.expand_as(output))

    def quantize(x, *args):
        return (torch.zeros_like(x, dtype=torch.float8_e4m3fn),
                torch.zeros((x.shape[0], x.shape[1] // 32), dtype=torch.uint8, device=x.device))

    monkeypatch.setattr(module, "_MOE_FUSED", {})
    monkeypatch.setattr(native, "kernel", lambda library, name: name)
    monkeypatch.setattr(native, "BoundLaunch", lambda name, *args: act if name.endswith("_act") else down)
    monkeypatch.setattr(quant, "act_quant", quantize)
    x = torch.ones((1, 3, 256), dtype=torch.bfloat16, device="cuda")
    try:
        if kernel_failure:
            with pytest.raises(RuntimeError, match="expert launch failed"):
                layer.forward(x)
        else:
            result = layer.forward(x)
            expected = (ids.float() * weights).sum(dim=1) + 2
            torch.testing.assert_close(result[0, :, 0].float(), expected)
            assert layer._last_ids == [1, 2, 3, 5, 6]
            assert mgr.stats["queue_layers"] == 2
        assert not any(cache._refs)
        assert not cache._urgent
    finally:
        cache.synchronize()


def test_unsplit_moe_capture_is_rejected_before_supply(monkeypatch):
    from tinyinfer.deepseek.moe import MoE

    layer = MoE.__new__(MoE)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    value = torch.zeros((1, 32))
    with pytest.raises(RuntimeError, match="external route-resolution"):
        layer._forward_gpu_only(value, value, torch.zeros((1, 1), dtype=torch.long), value)


def test_copy_cannot_overwrite_a_published_reader():
    cache = pool(1)
    cache.ensure(0, [7], source)
    index = cache.acquire_ready(0, [7])[0]
    with pytest.raises(RuntimeError, match="unpublished reservation"):
        cache.copy_in(index, source(0, [9])[9])
    assert bool((cache.buffer(index) == 7).all())
    cache.release(0, 7)


def test_batch_rejects_duplicate_identities_without_partial_leases():
    cache = pool(2)
    cache.ensure(0, [7], source)
    with pytest.raises(ValueError, match="unique expert identities"):
        cache.acquire_many(0, [7, 7])
    assert not any(cache._refs)


def test_failed_consumer_event_keeps_every_reader_lease():
    cache = pool(2)
    cache.ensure(0, [1, 2], source)
    cache.acquire_ready(0, [1, 2])
    cache._buffers = [SimpleNamespace(device=torch.device("cuda:0")),
                      SimpleNamespace(device=torch.device("cuda:1"))]
    events = [SimpleNamespace(record=Mock()),
              SimpleNamespace(record=Mock(side_effect=RuntimeError("record failed")))]
    cache._torch = SimpleNamespace(cuda=SimpleNamespace(
        Event=lambda: events.pop(0),
        current_stream=lambda device: SimpleNamespace(cuda_stream=device.index + 1)))
    with pytest.raises(RuntimeError, match="record failed"):
        cache.release_many(0, [1, 2])
    assert cache._refs == [1, 1]
    assert cache._consumer_events == [{}, {}]


def test_source_retirement_error_does_not_repeat_successful_releases():
    cache = pool(2)
    mgr, refs = source_manager(cache)
    refs[(0, 1)] = refs[(0, 2)] = 1
    failed = SimpleNamespace(query=Mock(side_effect=RuntimeError("query failed")))
    mgr._source_flights = [((0, 1), None), ((0, 2), failed)]
    with pytest.raises(RuntimeError, match="query failed"):
        mgr.drain_sources()
    assert refs[(0, 1)] == 0
    assert refs[(0, 2)] == 1
    failed.query = lambda: True
    mgr.drain_sources()
    assert all(value == 0 for value in refs.values())

@pytest.mark.parametrize("experts", [[7], [7, 9]])
def test_fetch_failure_releases_all_acquired_slots_and_host_leases(experts):
    cache = pool(2)
    if len(experts) > 1:
        cache.ensure(0, experts, source)
    store = ExpertStore.__new__(ExpertStore)
    store.slots = cache
    store._freq = defaultdict(int)
    store._open = defaultdict(int)
    store._lock = threading.Lock()
    refs = defaultdict(int)

    def lease(key, delta):
        refs[key] += delta
        assert refs[key] >= 0

    store.ram = SimpleNamespace(lease=lease, contains_many=lambda layer, wanted: {e: True for e in wanted})
    produced = []

    def materialize(layer, expert, **kwargs):
        assert refs[(layer, expert)] == 1
        if expert == experts[-1]:
            raise RuntimeError("materialization failed")
        record = SimpleNamespace(key=(layer, expert), event=None)
        produced.append(record)
        return record

    store._materialize = materialize
    with pytest.raises(RuntimeError, match="materialization failed"):
        store.fetch(0, experts)
    assert not any(cache._refs)
    assert not cache._urgent
    assert not any(store._open.values())
    assert not any(refs.values())
    assert len(produced) == len(experts) - 1


def test_record_release_waits_for_pending_source_dma():
    store = ExpertStore.__new__(ExpertStore)
    store.slots = pool(1)
    store.slots.ensure(0, [7], source)
    store.slots.acquire_ready(0, [7])
    store._lock = threading.Lock()
    store._open = defaultdict(int, {(0, 7): 1})
    event = PendingEvent()

    def unlease(key, delta):
        assert event.done
        assert key == (0, 7) and delta == -1

    store.ram = SimpleNamespace(lease=unlease)
    store.release([SimpleNamespace(key=(0, 7), event=event)])
    assert store.slots._refs == [0]
    assert store._open[(0, 7)] == 0

