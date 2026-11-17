"""主机源、缓存、保留区与实际池申请必须使用同一份规划。"""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tinyinfer.engine import Engine
from tinyinfer.planner import Layout
from tinyinfer.resources import model_profile
from test_weight_ring_plan import planner


def test_host_weights_use_exact_remaining_layers():
    p = planner()
    p.model["layer_bytes_by_id"] = {0: 64, 1: 128, 2: 256, 3: 512}
    lay = Layout(resident=1)
    assert p._plan_host_pool(lay)
    assert lay.host_weight_bytes == 128 + 256 + 512
    assert lay.host_pool_bytes + lay.host_reserve_bytes <= p.model["host_available_bytes"]
    assert lay.cache_bytes <= p.expert_b * p.n_layers * p.model["n_experts"]


def test_dense_model_gets_host_weights_without_an_expert_cache():
    p = planner(experts=False)
    lay = Layout(resident=1)
    assert p._plan_host_pool(lay)
    assert lay.host_weight_bytes > 0
    assert lay.cache_bytes == 0
    assert lay.host_pool_bytes == lay.host_weight_bytes + p.max_layer_b


def test_unknown_layer_sizes_are_not_replaced_with_an_average():
    p = planner()
    p.model.pop("layer_bytes_by_id")
    with pytest.raises(RuntimeError, match="per-layer byte sizes"):
        p._plan_host_pool(Layout(resident=1))


def test_host_shortage_rejects_the_layout():
    p = planner()
    p.model["host_available_bytes"] = 0
    assert not p._plan_host_pool(Layout(resident=1))


def test_gpu_slots_do_not_charge_host_cache_a_second_time():
    p = planner()
    first, second = Layout(resident=1, slots=2), Layout(resident=1, slots=8)
    assert p._plan_host_pool(first) and p._plan_host_pool(second)
    assert first.host_pool_bytes == second.host_pool_bytes
    assert first.cache_bytes == second.cache_bytes


def test_runtime_claims_exact_plan_ignoring_old_size_overrides(monkeypatch):
    import tinyinfer.budget as budget_module

    allocator = SimpleNamespace(_claim_buf=bytearray(64), set_limit=Mock(),
                                claim=Mock(return_value=64))
    monkeypatch.setattr(budget_module, "budget", lambda: allocator)
    monkeypatch.setenv("TI_POOL_RATIO", "0.01")
    monkeypatch.setenv("TI_HOST_POOL_GB", "99")
    eng = Engine.__new__(Engine)
    eng.hw_plan = Layout(host_pool_bytes=64, host_reserve_bytes=32)
    eng._model_prof = {"host_available_bytes": 96}
    result = eng._allocate_host_pool()
    assert result.total == 64
    allocator.claim.assert_called_once_with(64)
    allocator.set_limit.assert_called_once_with(96)


def test_runtime_rejects_changed_capacity_without_shrinking(monkeypatch):
    import tinyinfer.budget as budget_module

    allocator = SimpleNamespace(set_limit=Mock(), claim=Mock())
    monkeypatch.setattr(budget_module, "budget", lambda: allocator)
    eng = Engine.__new__(Engine)
    eng.hw_plan = Layout(host_pool_bytes=64, host_reserve_bytes=32)
    eng._model_prof = {"host_available_bytes": 95}
    with pytest.raises(RuntimeError, match="planned against"):
        eng._allocate_host_pool()
    allocator.claim.assert_not_called()


def test_profile_exports_aligned_per_layer_sizes(tmp_path):
    entries = {"layers.0.a": {"dtype": "U8", "data_offsets": [0, 3]},
               "layers.0.b": {"dtype": "U8", "data_offsets": [3, 10]},
               "layers.1.a": {"dtype": "U8", "data_offsets": [10, 19]}}
    index = SimpleNamespace(weight_map={name: "shard" for name in entries},
                            _header=lambda shard: (entries, 0),
                            model_dir=tmp_path, expert_blob_bytes=lambda: 0)
    profile = model_profile(index)
    assert profile["layer_bytes_by_id"] == {0: 512, 1: 256}
    assert profile["max_layer_bytes"] == 512


def test_dense_profile_does_not_ask_for_a_nonexistent_expert(tmp_path):
    entries = {"layers.0.a": {"dtype": "U8", "data_offsets": [0, 16]}}
    index = SimpleNamespace(weight_map={name: "shard" for name in entries},
                            _header=lambda shard: (entries, 0), model_dir=tmp_path,
                            expert_blob_bytes=Mock(side_effect=AssertionError("no experts")))
    assert model_profile(index)["expert_bytes"] == 0
    index.expert_blob_bytes.assert_not_called()


def test_broken_expert_metadata_is_not_replaced_by_a_fixed_size():
    from tinyinfer.loading import ShardIndex

    index = ShardIndex.__new__(ShardIndex)
    index.expert_blob = Mock(side_effect=KeyError("missing expert tensor"))
    with pytest.raises(KeyError, match="missing expert tensor"):
        index.expert_blob_bytes(2, 3)
    index.expert_blob.assert_called_once_with(2, 3)


def test_profile_uses_an_expert_that_actually_exists(tmp_path):
    entries = {"layers.0.a": {"dtype": "U8", "data_offsets": [0, 16]},
               "layers.2.ffn.experts.3.w1.weight": {"dtype": "U8", "data_offsets": [16, 32]}}
    index = SimpleNamespace(weight_map={name: "shard" for name in entries},
                            _header=lambda shard: (entries, 0), model_dir=tmp_path,
                            expert_blob_bytes=Mock(return_value=1234))
    assert model_profile(index)["expert_bytes"] == 1234
    index.expert_blob_bytes.assert_called_once_with(2, 3)
