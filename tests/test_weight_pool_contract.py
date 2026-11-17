"""已指定主机池时，权重读取不能私自改用另一块内存。"""
import mmap
import struct

import pytest
import torch

from tinyinfer.budget import HostPool
from tinyinfer.weights import ShardStore


def store(tmp_path, capacity=4096, content=struct.pack("<4f", 1, 2, 3, 4)):
    obj = ShardStore.__new__(ShardStore)
    obj.model_dir = tmp_path
    obj.build_pool = HostPool.from_claimed(mmap.mmap(-1, capacity), capacity)
    obj.shard_of = lambda name: "weights.bin"
    obj._header = lambda shard: ({"weight": {"data_offsets": [0, 16], "dtype": "F32", "shape": [4]}}, 0)
    (tmp_path / "weights.bin").write_bytes(content)
    return obj


def test_reads_into_assigned_pool_and_exposes_reclamation_identity(tmp_path):
    obj = store(tmp_path)
    value = obj.get("weight")
    assert torch.equal(value, torch.tensor([1., 2., 3., 4.]))
    assert value.data_ptr() == obj.build_pool._base + value._tiny_pool_off
    assert obj.build_pool.used == value._tiny_pool_len == 16


def test_pool_exhaustion_does_not_read_into_private_memory(tmp_path):
    obj = store(tmp_path, capacity=8)
    with pytest.raises(MemoryError, match="planned host pool"):
        obj.get("weight")
    assert obj.build_pool.used == 0


def test_truncated_read_returns_slice_to_pool(tmp_path):
    obj = store(tmp_path, content=b"short")
    with pytest.raises(ValueError, match="truncated"):
        obj.get("weight")
    assert obj.build_pool.used == 0


def test_bad_tensor_shape_returns_slice_to_pool(tmp_path):
    obj = store(tmp_path)
    obj._header = lambda shard: ({"weight": {"data_offsets": [0, 16], "dtype": "F32", "shape": [5]}}, 0)
    with pytest.raises(RuntimeError):
        obj.get("weight")
    assert obj.build_pool.used == 0
