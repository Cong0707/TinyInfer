"""会话 KV 的换入换出：卡上只放活跃的，其余沉到内存。

这是并发能力的关键。一条 1M 会话的压缩 KV 是 6.9GB，44GB 卡扣掉权重/激活/
槽位后只剩约 30GB —— 不换出的话并发上限就是 4。内存 371GB 放得下 50 多条。
"""

import pytest

torch = pytest.importorskip("torch")

from tinyinfer.kv_tiers import KvSwapPool, to_host  # noqa: E402


def _snap(nbytes: int, device: str = "cpu"):
    return {"kv": torch.zeros(nbytes // 2, dtype=torch.bfloat16, device=device)}


def test_put_get_roundtrip():
    pool = KvSwapPool(1 << 30)
    snap = _snap(4096)
    pool.put("s1", snap)
    assert pool.get("s1") is snap
    st = pool.stats()
    assert st["entries"] == 1 and st["swapped_out"] == 1 and st["swapped_in"] == 1


def test_capacity_drops_least_recently_used_session():
    """超容量时丢的是整条会话，不是它内部的一段 —— 顺序性不会被破坏。"""
    pool = KvSwapPool(3 * 8192)          # 最多 3 条 8KB 会话
    for i in range(4):
        pool.put(f"s{i}", _snap(8192))
    assert pool.get("s0") is None, "the oldest session should have been dropped"
    assert pool.get("s3") is not None
    assert pool.stats()["dropped"] >= 1


def test_get_refreshes_recency():
    pool = KvSwapPool(3 * 8192)
    for i in range(3):
        pool.put(f"s{i}", _snap(8192))
    pool.get("s0")                       # 让它变成最近用过
    pool.put("s3", _snap(8192))          # 必须挤掉别人的位置
    assert pool.get("s0") is not None
    assert pool.get("s1") is None


def test_to_host_moves_every_tensor():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    snap = {"kv": torch.zeros(64, device="cuda"),
            "ckv": torch.zeros(32, device="cuda"),
            "pending": (torch.zeros(8, device="cuda"), torch.zeros(8, device="cuda"))}
    host = to_host(snap)
    assert host["kv"].device.type == "cpu"
    assert host["ckv"].device.type == "cpu"
    assert host["pending"][0].device.type == "cpu"


def test_drop_removes_one_session():
    pool = KvSwapPool(1 << 30)
    pool.put("a", _snap(1024))
    assert pool.drop("a") is True
    assert pool.get("a") is None
    assert pool.drop("a") is False
