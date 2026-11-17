"""Use real FP4 kernels with tiny synthetic expert weights, without an Engine."""
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from tinyinfer.deepseek import moe as module
from tinyinfer.expert_tiers import ExpertTierManager
from tinyinfer.loading import VramSlotCache


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_real_expert_kernels_agree_across_slot_capacity_and_route_changes(monkeypatch):
    dim = mid = 256
    packed = mid * dim // 2
    scales = mid * dim // 32
    stride = packed + scales
    blob = 3 * stride
    offsets = (0, packed, stride, stride + packed, 2 * stride, 2 * stride + packed)
    host_weights = {}
    for expert in (1, 2, 3, 5, 6):
        weight = torch.empty(blob, dtype=torch.uint8, pin_memory=True)
        for base in (0, stride):
            weight[base:base + packed].fill_(0x22)
            weight[base + packed:base + stride].fill_(127)
        code = expert % 3 + 1
        weight[2 * stride:2 * stride + packed].fill_(code | (code << 4))
        weight[2 * stride + packed:].fill_(127)
        host_weights[expert] = weight
    routes = torch.tensor([[1, 2], [2, 3], [5, 6]], dtype=torch.long, device="cuda")
    route_weights = torch.tensor([[0.25, 0.75], [0.5, 0.5], [0.375, 0.625]], device="cuda")
    x = torch.full((3, dim), 1 / dim, dtype=torch.bfloat16, device="cuda")
    monkeypatch.setattr(module, "_MOE_FUSED", {})

    def execute(capacity):
        slots = VramSlotCache(capacity, blob, device="cuda", slot_arena=True)
        store = SimpleNamespace(slots=slots, ram=None, reader=None, index=None)
        mgr = ExpertTierManager(store, n_layers=1, n_experts=7, warm=False)
        mgr._provide = lambda layer, wanted: {e: memoryview(host_weights[e].numpy()) for e in wanted}
        layer = module.MoE.__new__(module.MoE)
        layer.cfg = SimpleNamespace(dim=dim, moe_inter_dim=mid, n_routed_experts=7)
        layer.layer_id = 0
        layer._expert_store = store
        layer._mgr = mgr
        layer._route_counts = {}
        layer._ts_offsets = lambda _: offsets
        layer.swiglu_limit = 0.0
        try:
            first = layer._forward_gpu_only(x, x, routes, route_weights).clone()
            torch.cuda.synchronize()
            changed = routes.flip(0).contiguous()
            second = layer._forward_gpu_only(x, x, changed, route_weights).clone()
            torch.cuda.synchronize()
            assert not any(slots._refs)
            assert not slots._urgent
            return first.cpu(), second.cpu()
        finally:
            slots.synchronize()
            mgr.close()

    whole = execute(5)
    partitioned = execute(3)
    for expected, actual in zip(whole, partitioned):
        assert bool(torch.isfinite(actual).all())
        assert bool((actual.abs() > 0).all())
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def reference(ids):
        result = torch.zeros((ids.shape[0], dim), dtype=torch.float32)
        for row in range(ids.shape[0]):
            for pos, expert in enumerate(ids[row].tolist()):
                routed = (F.silu(torch.ones(mid)) * route_weights[row, pos].cpu()).to(torch.bfloat16)
                amax = routed.float().abs().amax().clamp_min(1e-4)
                scale = torch.pow(2.0, torch.ceil(torch.log2(amax / torch.finfo(torch.float8_e4m3fn).max)))
                quantized = (routed.float() / scale).to(torch.float8_e4m3fn).float() * scale
                down_weight = (expert % 3 + 1) / 2
                contribution = (quantized.sum() * down_weight).to(torch.bfloat16)
                result[row] += contribution.float()
        return result.to(torch.bfloat16)

    torch.testing.assert_close(partitioned[0], reference(routes.cpu()), rtol=0, atol=0)
    torch.testing.assert_close(partitioned[1], reference(routes.flip(0).cpu()), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(("dim", "mid"), [(128, 256), (256, 128)])
def test_unsupported_reduction_is_rejected_before_allocating_slots(dim, mid):
    layer = module.MoE.__new__(module.MoE)
    layer.cfg = SimpleNamespace(dim=dim, moe_inter_dim=mid)
    layer.layer_id = 0
    value = torch.zeros((1, dim), dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="multiples of 256"):
        layer._forward_gpu_only(value, value, torch.zeros((1, 1), dtype=torch.long, device="cuda"),
                                torch.ones((1, 1), device="cuda"))
