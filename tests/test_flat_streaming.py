"""无需加载模型的权重环回归：复制流、消费事件、跨步复用。"""
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from tinyinfer.engine import Engine
from tinyinfer.deepseek.model import Block


class Stream:
    def __init__(self):
        self.waited = []

    def wait_event(self, event):
        self.waited.append(event)

    def wait_stream(self, stream):
        self.waited.append(stream)


class Event:
    def record(self, stream):
        self.stream = stream


@pytest.fixture
def cuda(monkeypatch):
    compute = Stream()
    state = SimpleNamespace(current=compute)

    @contextmanager
    def scope(stream):
        previous = state.current
        state.current = stream
        try:
            yield
        finally:
            state.current = previous

    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "stream", scope)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *args, **kwargs: state.current)
    monkeypatch.setattr(torch.cuda, "Stream", lambda *args, **kwargs: Stream())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    return compute


def make_engine(monkeypatch, *, layers=7, ring=3):
    eng = Engine.__new__(Engine)
    eng.device = "cuda"
    eng._copy_stream = Stream()
    eng._flat_ev = {}
    blocks, layout, host = {}, {}, {}
    calls = []
    for layer in range(layers):
        tensor = torch.full((2,), float(layer + 1))
        block = SimpleNamespace(weight=tensor, engram=None, _device="cpu")
        block._weight_paths = lambda b=block: [(b, "weight", b.weight)]
        block._cpu_refs = {Block._wkey(block, "weight"): tensor}
        blocks[layer] = block
        size = tensor.numel() * tensor.element_size()
        layout[layer] = ([(0, size)], size)
        host[layer] = tensor.view(torch.uint8)
    bufs = [torch.empty(size, dtype=torch.uint8) for _ in range(ring)]
    eng._flat_state = (layout, bufs, host)
    eng.block = blocks.__getitem__

    def dma(dst, src, size, stream=None):
        assert stream is eng._copy_stream
        calls.append((stream, size))
        dst.view(torch.uint8).reshape(-1)[:size].copy_(src.view(torch.uint8).reshape(-1)[:size])
        return True

    monkeypatch.setattr("tinyinfer.loading._async_h2d", dma)
    return eng, calls


@pytest.mark.parametrize("layers,ring", [(7, 1), (7, 3), (1, 4), (5, 2)])
def test_one_copy_per_layer_across_steps(monkeypatch, cuda, layers, ring):
    eng, calls = make_engine(monkeypatch, layers=layers, ring=ring)
    for _ in range(6):
        for layer in range(layers):
            eng._flat_wait(layer, layer % ring)
            for ahead in range(layer + 1, min(layers, layer + ring)):
                if ahead not in eng._flat_ev:
                    eng._flat_prefetch(ahead, ahead % ring)
            assert torch.equal(eng.block(layer).weight, torch.full((2,), float(layer + 1)))
            done = Event()
            done.record(cuda)
            eng._flat_done[layer] = done
        assert not eng._flat_ev
    assert len(calls) == layers * 6
    assert eng._flat_applies == layers * 6
    assert all(event.stream is eng._copy_stream for event in cuda.waited)
    assert all(event.stream is cuda for event in eng._copy_stream.waited)


def test_duplicate_prefetch_is_rejected(monkeypatch, cuda):
    eng, calls = make_engine(monkeypatch)
    eng._flat_prefetch(0, 0)
    with pytest.raises(RuntimeError, match="duplicate"):
        eng._flat_prefetch(0, 0)
    assert len(calls) == 1


def test_unconsumed_slot_cannot_be_overwritten(monkeypatch, cuda):
    eng, calls = make_engine(monkeypatch)
    eng._flat_prefetch(0, 0)
    with pytest.raises(RuntimeError, match="has not finished"):
        eng._flat_prefetch(1, 0)
    assert len(calls) == 1


def test_wrong_slot_is_rejected(monkeypatch, cuda):
    eng, _ = make_engine(monkeypatch)
    eng._flat_prefetch(0, 0)
    with pytest.raises(RuntimeError, match="no longer matches"):
        eng._flat_wait(0, 1)


def test_dma_failure_does_not_switch_copy_path(monkeypatch, cuda):
    eng, _ = make_engine(monkeypatch)
    monkeypatch.setattr("tinyinfer.loading._async_h2d", lambda *args, **kwargs: False)
    with pytest.raises(RuntimeError, match="H2D submission failed"):
        eng._flat_prefetch(0, 0)
    assert not eng._flat_ev


def test_layout_failure_precedes_dma(monkeypatch, cuda):
    eng, calls = make_engine(monkeypatch)
    eng._flat_state[0][0] = ([], 8)
    with pytest.raises(RuntimeError, match="layout"):
        eng._flat_prefetch(0, 0)
    assert not calls


def test_capture_does_not_enqueue_host_copy(monkeypatch, cuda):
    eng, calls = make_engine(monkeypatch)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    eng._flat_wait(0, 0)
    assert not calls
    assert not cuda.waited


def test_removed_layer_prefetch_has_no_entrypoint():
    for name in ("_layer_prefetch", "_ensure_stream_buffers", "_stream_copy_sync",
                 "_stream_prefetch", "_ensure_slots", "_slot_for", "_stream_into_slot"):
        assert not hasattr(Engine, name)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_real_cuda_ring_on_nondefault_compute_stream():
    eng = Engine.__new__(Engine)
    eng.device = "cuda"
    eng._copy_stream = torch.cuda.Stream()
    eng._flat_ev = {}
    blocks, host, layout = {}, {}, {}
    for layer in range(7):
        tensor = torch.full((32 + layer,), float(layer + 1), pin_memory=True)
        block = SimpleNamespace(weight=tensor, engram=None, _device="cpu")
        block._weight_paths = lambda b=block: [(b, "weight", b.weight)]
        block._cpu_refs = {Block._wkey(block, "weight"): tensor}
        blocks[layer] = block
        size = tensor.numel() * tensor.element_size()
        layout[layer] = ([(0, size)], size)
        host[layer] = tensor.view(torch.uint8)
    bufs = [torch.empty(size, dtype=torch.uint8, device="cuda") for _ in range(3)]
    eng._flat_state = (layout, bufs, host)
    eng.block = blocks.__getitem__
    eng._copy_stream.wait_stream(torch.cuda.current_stream())
    compute = torch.cuda.Stream()
    compute.wait_stream(torch.cuda.current_stream())
    outputs = []
    with torch.cuda.stream(compute):
        for _ in range(6):
            for layer in blocks:
                eng._flat_wait(layer, layer % len(bufs))
                for ahead in range(layer + 1, min(len(blocks), layer + len(bufs))):
                    if ahead not in eng._flat_ev:
                        eng._flat_prefetch(ahead, ahead % len(bufs))
                outputs.append((layer, blocks[layer].weight.clone()))
                done = torch.cuda.Event()
                done.record(compute)
                eng._flat_done[layer] = done
    compute.synchronize()
    for layer, value in outputs:
        assert torch.equal(value.cpu(), torch.full((32 + layer,), float(layer + 1)))
    assert eng._flat_applies == len(blocks) * 6


def make_host_engine():
    from tinyinfer.budget import HostPool
    from tinyinfer.planner import Layout

    eng = Engine.__new__(Engine)
    eng.device = "cpu"
    eng.host_pool = HostPool.from_claimed(bytearray(4096), 4096)
    eng.hw_plan = Layout(weight_buffers=2, weight_buffer_bytes=256, host_weight_bytes=768)
    eng._stream_layers = lambda: {0, 1, 2}
    blocks = {}
    for layer in range(3):
        raw = eng.host_pool.tensor(256, align=256)
        tensor = raw.view(torch.float32).fill_(layer + 1)
        tensor._tiny_pool_off = raw._tiny_pool_off
        tensor._tiny_pool_len = raw._tiny_pool_len
        block = SimpleNamespace(weight=tensor, engram=None, _device="cpu")
        block._weight_paths = lambda b=block: [(b, "weight", b.weight)]
        blocks[layer] = block
    eng.block = blocks.__getitem__
    return eng


def test_host_packing_reclaims_sources_and_respects_planned_quota(cuda):
    eng = make_host_engine()
    eng._flat_setup()
    assert eng.host_pool.used == 768
    assert len(eng._flat_state[1]) == 2
    assert set(eng._flat_state[2]) == {0, 1, 2}
    for layer in range(3):
        block = eng.block(layer)
        assert torch.equal(block.weight, torch.full((64,), float(layer + 1)))
        assert block.weight.data_ptr() == eng._flat_state[2][layer].data_ptr()


def test_host_packing_short_quota_fails_before_allocation(cuda):
    eng = make_host_engine()
    eng.hw_plan.host_weight_bytes -= 1
    with pytest.raises(RuntimeError, match="host weights need"):
        eng._flat_setup()
    assert eng.host_pool.used == 768
    assert not hasattr(eng, "_flat_state")


def test_host_copy_failure_does_not_publish_partial_layout(cuda, monkeypatch):
    eng = make_host_engine()
    def fail(*args, **kwargs):
        raise RuntimeError("copy failed")
    monkeypatch.setattr(torch.Tensor, "copy_", fail)
    with pytest.raises(RuntimeError, match="copy failed"):
        eng._flat_setup()
    assert eng.host_pool.used == 768
    assert not hasattr(eng, "_flat_state")


@pytest.mark.parametrize("host", [None, {}])
def test_missing_slab_does_not_switch_to_tensor_sources(monkeypatch, cuda, host):
    eng, calls = make_engine(monkeypatch)
    layout, buffers, _ = eng._flat_state
    eng._flat_state = (layout, buffers, host)
    with pytest.raises(RuntimeError, match="no planned host slab"):
        eng._flat_prefetch(0, 0)
    assert not calls
