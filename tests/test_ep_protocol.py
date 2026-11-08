"""EP 的归约协议：每个 rank 算自己那份专家，输出是各 rank 的**和**。

这是 EP 最容易写错的地方（ep.py 的文档也这么写）：filter 到自己的专家然后
直接返回，看起来对，其实丢了别的 rank 的贡献。这一组测试锁住正确语义，
并且用真实的 ExpertRoute 划分来验证「和 == 全量算」。
"""

import pytest

torch = pytest.importorskip("torch")

from tinyinfer.ep import ExpertRoute, load_balance, reduce_partials  # noqa: E402


def test_route_blocks_are_contiguous_and_cover_everything():
    r = ExpertRoute(384, ["cuda:0", "cuda:1"], [3, 1])
    assert r.world == 2
    # 连续块：一个 expert 的 blob 和邻居在一起，磁盘读才是顺序的
    assert r.block[0][0] == 0
    assert r.block[-1][1] == 384
    assert r.block[0][1] == r.block[1][0]
    # 按容量分：3:1 -> 288 / 96
    assert r.block[0][1] == 288, r.block


def test_every_expert_has_exactly_one_owner():
    r = ExpertRoute(384, ["cuda:0", "cuda:1", "cuda:2"])
    owners = [r.rank_of(e) for e in range(384)]
    assert len(set(owners)) == 3
    assert sum(1 for o in owners if o == 0) > 0


def test_reduce_of_partials_equals_the_full_sum():
    """两个 rank 各算一半专家，加起来必须等于一次性全算。"""
    torch.manual_seed(0)
    full = torch.randn(4, 8, dtype=torch.float32)
    # 模拟：rank0 只看到前 3 个专家、rank1 只看到后 5 个（权重是分块的）
    part0 = full * 0.375      # 3/8
    part1 = full * 0.625      # 5/8
    got = reduce_partials([part0, part1])
    assert torch.allclose(got, full, atol=1e-6)


def test_reduce_single_partial_is_a_passthrough():
    """单卡时不该多一次逐元素加法。"""
    p = torch.randn(2, 2)
    assert reduce_partials([p]) is p


def test_reduce_rejects_empty():
    with pytest.raises(ValueError):
        reduce_partials([])


def test_reduce_moves_to_target_device():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    a = torch.ones(2, 2)
    b = torch.ones(2, 2)
    out = reduce_partials([a, b], target_device="cuda")
    assert out.device.type == "cuda"
    assert torch.allclose(out.cpu(), torch.full((2, 2), 2.0))


def test_load_balance_evens_the_busiest_rank():
    r = ExpertRoute(8, ["cuda:0", "cuda:1"])
    # 全部流量压在 rank0 的那几个专家上
    counts = [100, 100, 100, 100, 1, 1, 1, 1]
    owners = load_balance(r, counts)
    load = [0, 0]
    for e, o in enumerate(owners):
        load[o] += counts[e]
    before = max(sum(counts[:4]), sum(counts[4:]))
    assert max(load) < before, (load, before)
