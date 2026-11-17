"""规划预算与权重环实际分配必须同源，不启动完整引擎。"""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tinyinfer.engine import Engine
from tinyinfer.planner import Layout, Planner
from tinyinfer.resources import Card, Hardware

GB = 1 << 30


def planner(vram=2 * GB, *, experts=True, compute=0.003):
    hw = Hardware(name="synthetic", vram_gb=vram / GB, ram_gb=8,
                  ram_avail_gb=6, cards=[Card(0, "synthetic", 8, vram / GB)])
    model = {"n_layers": 4, "layer_bytes": 256 << 20, "max_layer_bytes": 512 << 20,
             "embed_bytes": 128 << 20, "head_bytes": 128 << 20,
             "expert_bytes": (8 << 20) if experts else 0, "topk": 2 if experts else 0,
             "n_experts": 8 if experts else 0, "max_position_embeddings": 128,
             "rotary_dim": 32, "layer_bytes_by_id": {i: 256 << 20 for i in range(4)},
             "host_available_bytes": 6 * GB}
    bw = {"h2d_bps": 7e9, "disk_bps": 4e9, "disk_rand_bps": 2e9, "ram_bps": 30e9}
    cal = {"runtime_overhead_bytes": 64 << 20, "route_reuse": 0.2,
           "h2d_layer_s": model["layer_bytes"] / bw["h2d_bps"],
           "h2d_expert_s": model["expert_bytes"] / bw["h2d_bps"]}
    if compute is not None:
        cal["compute_per_layer_s"] = compute
    return Planner(hw, model, bandwidth=bw, calibrate=cal)


def test_ring_bytes_are_counted_exactly():
    p = planner()
    without = Layout(resident=1, slots=2)
    with_ring = Layout(resident=1, slots=2, weight_buffers=3, weight_buffer_bytes=512 << 20)
    assert p.used_bytes(with_ring) - p.used_bytes(without) == 3 * (512 << 20)
    assert not p.fits(without)


@pytest.mark.parametrize("experts", [True, False])
def test_solver_produces_valid_pool_sizes_without_querying_cuda(monkeypatch, experts):
    monkeypatch.setattr(torch.cuda, "mem_get_info", Mock(side_effect=AssertionError("live CUDA query")))
    p = planner(vram=3 * GB // 2, experts=experts)
    lay = p.solve()
    assert p.fits(lay)
    assert 1 <= lay.weight_buffers <= lay.stream_layers
    assert lay.weight_buffer_bytes == max(p._layer_bytes()[lay.resident:])
    assert lay.slots >= p.topk if experts else lay.slots == 0


def test_fully_resident_layout_does_not_allocate_a_weight_ring():
    p = planner(vram=8 * GB)
    lay = p.solve()
    assert lay.resident == p.n_layers
    assert lay.weight_buffers == lay.weight_buffer_bytes == 0


def test_missing_compute_measurement_does_not_invent_overlap_benefit():
    p = planner(vram=3 * GB // 2, compute=None)
    lay = p.solve()
    assert lay.weight_buffers == 1


def test_too_small_device_is_rejected():
    with pytest.raises(RuntimeError, match="no feasible layout"):
        planner(vram=1).solve()


@pytest.mark.parametrize("count", [1, 2, 3])
def test_runtime_allocates_exact_planned_bytes(monkeypatch, count):
    monkeypatch.setenv("TI_RING", "999")
    eng = Engine.__new__(Engine)
    eng.device = "cpu"
    eng.hw_plan = Layout(weight_buffers=count, weight_buffer_bytes=32)
    buffers = eng._allocate_weight_ring({0: 8, 1: 16, 2: 24})
    assert len(buffers) == count
    assert sum(b.numel() * b.element_size() for b in buffers) == count * 32


def test_runtime_rejects_underestimated_geometry_before_allocating(monkeypatch):
    allocate = Mock(side_effect=AssertionError("must not allocate"))
    monkeypatch.setattr(torch, "empty", allocate)
    eng = Engine.__new__(Engine)
    eng.device = "cuda"
    eng.hw_plan = Layout(weight_buffers=1, weight_buffer_bytes=8)
    with pytest.raises(RuntimeError, match="planner reserved"):
        eng._allocate_weight_ring({0: 16})
    allocate.assert_not_called()


def test_setup_error_is_not_swallowed_or_retried(monkeypatch):
    monkeypatch.setenv("TI_FLAT", "0")
    eng = Engine.__new__(Engine)
    eng.device = "cuda"
    eng._stream_layers = Mock(return_value={0})
    error = RuntimeError("planned allocation failed")
    eng._flat_setup = Mock(side_effect=error)
    with pytest.raises(RuntimeError) as caught:
        eng._flat_enabled()
    assert caught.value is error
    eng._flat_setup.assert_called_once()
    assert not hasattr(eng, "_flat_state")


def test_resident_bytes_follow_the_actual_prefix_not_the_mean():
    p = planner()
    p.model["layer_bytes_by_id"] = {0: 900, 1: 100, 2: 100, 3: 100}
    before = Layout(resident=0)
    after = Layout(resident=1)
    assert p.used_bytes(after) - p.used_bytes(before) == 900


def test_ring_does_not_reserve_space_for_a_resident_heavy_layer():
    p = planner()
    p.model["layer_bytes_by_id"] = {0: 512 << 20, 1: 32 << 20, 2: 32 << 20, 3: 32 << 20}
    lay = Layout(resident=1, slots=2, weight_buffers=1, weight_buffer_bytes=32 << 20,
                 embed_on_cpu=True, head_on_cpu=True)
    assert p.fits(lay)


def test_heterogeneous_layer_pipeline_can_benefit_from_a_deeper_ring():
    p = planner()
    p.n_layers = 3
    p.layer_b = 1
    p.model["layer_bytes_by_id"] = {0: 1, 1: 1, 2: 10}
    p.cal.update(h2d_layer_s=1.0, compute_per_layer_s=5.0)
    assert p._weight_stream_seconds(0, 1) == 12
    assert p._weight_stream_seconds(0, 2) == 6
    assert p._weight_stream_seconds(0, 3) == 2
