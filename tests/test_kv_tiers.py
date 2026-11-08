import pytest

from tinyinfer.kv_tiers import KvTierPlan


def test_kv_budget_is_free_minus_reserve():
    plan = KvTierPlan(vram_free_bytes=30 * 2**30, reserve_bytes=10 * 2**30)
    assert plan.kv_budget() == 20 * 2**30


def test_groups_scale_with_budget():
    small = KvTierPlan(vram_free_bytes=8 * 2**30, reserve_bytes=4 * 2**30)
    big = KvTierPlan(vram_free_bytes=80 * 2**30, reserve_bytes=20 * 2**30)
    n_small = small.groups_that_fit(512)
    n_big = big.groups_that_fit(512)
    assert n_big > n_small
    assert n_big / n_small == pytest.approx(60 / 4, rel=0.01)


def test_zero_budget_is_safe():
    assert KvTierPlan(vram_free_bytes=0).kv_budget() == 0
    assert KvTierPlan(vram_free_bytes=100, reserve_bytes=200).kv_budget() == 0


def test_1m_context_kv_is_several_gb():
    """按这个 checkpoint 的真实 compress_ratios 算 1M 上下文的压缩 KV。

    实测配置（probe_kv2.py）：0 x 5 层、2 x 18 层、1 x 20 层，head_dim=512，
    latent 以 fp4 存（2 值/字节）。probe_kv.py 独立算出来是 6.92 GB，
    和这里必须一致 —— 两个算法互相对照，任一处算错都会被这条测出来。
    """
    head_dim = 512
    ratios = [0] * 5 + [2] * 18 + [1] * 20
    total = 0
    for r in ratios:
        if r > 0:
            total += (1_000_000 // r) * head_dim // 2
    assert total / 2**30 == pytest.approx(6.9, rel=0.05)
    # 一条 1M 会话的 KV ~6.9GB。30GB 预算下同时能驻留的会话数就是这个比值 ——
    # 这才是「并发能力」的真实数字，也是为什么必须做 KV 分层。
    plan = KvTierPlan(vram_free_bytes=30 * 2**30)
    concurrent = plan.kv_budget() // total
    assert concurrent == 4, concurrent
    # 内存放得下几十条：不活跃的沉到内存，活跃的留在卡上
    assert (371 * 2**30) // total > 30

