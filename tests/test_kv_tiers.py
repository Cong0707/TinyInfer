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




def test_plan_slots_scales_with_the_card():
    """同一个函数在 8GB / 44GB / 141GB 上算出的并发数必须不同。"""
    from tinyinfer.kv_tiers import plan_slots

    n8 = plan_slots(vram_free_bytes=7 * 2**30)
    n44 = plan_slots(vram_free_bytes=42 * 2**30)
    n141 = plan_slots(vram_free_bytes=138 * 2**30)
    assert n8 == 1, n8
    assert 4 <= n44 <= 7, n44
    assert n141 > n44, (n44, n141)


def test_plan_slots_never_returns_zero():
    from tinyinfer.kv_tiers import plan_slots

    assert plan_slots(vram_free_bytes=0) >= 1
    assert plan_slots(vram_free_bytes=1 * 2**30) >= 1


def test_plan_slots_shrinks_with_context():
    """上下文越长，同时驻留的会话越少 —— 这是长上下文和并发的真实 trade-off。"""
    from tinyinfer.kv_tiers import plan_slots

    free = 42 * 2**30
    n_1m = plan_slots(vram_free_bytes=free, context_tokens=1_048_576)
    n_128k = plan_slots(vram_free_bytes=free, context_tokens=131_072)
    assert n_128k > n_1m, (n_128k, n_1m)



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


def test_union_estimate_hits_the_measured_points():
    """估算曲线必须精确穿过三个实测点。"""
    from tinyinfer.kv_tiers import union_estimate

    assert union_estimate(1) == 6.0
    assert abs(union_estimate(8) - 35.0) < 0.01
    assert abs(union_estimate(16) - 61.0) < 0.01
    # 单调递增，且不会超过专家总数
    prev = 0.0
    for n in range(1, 200):
        v = union_estimate(n)
        assert v >= prev, (n, v, prev)
        assert v <= 384, (n, v)
        prev = v


def test_slot_bound_matches_the_measured_union():
    """槽位约束算出来的批宽必须和实测并集一致。

    实测（tests/test_route_union.py）：8 条会话并集 35 个/层，40 层要 1400 个槽。
    L20 有 1359 -> 8 条放不下，7 条（约 30.7/层 x40 = 1228）放得下。
    """
    from tinyinfer.kv_tiers import batch_slots_bound

    n = batch_slots_bound(slots=1359, n_layers=40, topk=6)
    assert n == 7, n
    # 141GB 卡（约 5000 槽位）：能到几十条
    big = batch_slots_bound(slots=5000, n_layers=40, topk=6)
    assert big > 20, big


def test_small_pool_gives_one():
    """8GB 卡的 14 个槽位只够一条会话 —— 和 plan_slots 的结论一致。"""
    from tinyinfer.kv_tiers import batch_slots_bound

    assert batch_slots_bound(slots=14, n_layers=40, topk=6) == 1
    assert batch_slots_bound(slots=240, n_layers=40, topk=6) == 1


def test_zero_slots_is_safe():
    from tinyinfer.kv_tiers import batch_slots_bound

    assert batch_slots_bound(slots=0, n_layers=40, topk=6) == 1
    assert batch_slots_bound(slots=-5, n_layers=40, topk=6) == 1


def test_more_slots_never_reduce_the_bound():
    from tinyinfer.kv_tiers import batch_slots_bound

    prev = 0
    for slots in (100, 500, 1000, 2000, 5000, 20000):
        n = batch_slots_bound(slots=slots, n_layers=40, topk=6)
        assert n >= prev, (slots, n, prev)
        prev = n

def test_plan_expert_slots_single_card():
    """8GB 卡的 6.5GB 可用 -> 约 345 个槽位。"""
    from tinyinfer.kv_tiers import plan_expert_slots

    r = plan_expert_slots(
        devices=[{"name": "cuda:0", "free_bytes": 6_500_000_000}],
        expert_bytes=18_800_640)
    assert r["devices"] == 1
    assert 300 < r["slots"] < 400, r
    assert r["per_device"][0]["slots"] == r["slots"]


def test_plan_expert_slots_asymmetric_cards():
    """96GB + 32GB：大卡拿大头，但不能超过自己的余量。"""
    from tinyinfer.kv_tiers import plan_expert_slots

    r = plan_expert_slots(
        devices=[
            {"name": "cuda:0", "free_bytes": 90_000_000_000},
            {"name": "cuda:1", "free_bytes": 30_000_000_000},
        ],
        expert_bytes=18_800_640)
    a, b = r["per_device"][0]["slots"], r["per_device"][1]["slots"]
    assert a > b * 2, (a, b)
    assert r["slots"] == a + b


def test_plan_expert_slots_clamps_to_max():
    from tinyinfer.kv_tiers import plan_expert_slots

    r = plan_expert_slots(
        devices=[{"name": "cuda:0", "free_bytes": 6_500_000_000}],
        expert_bytes=18_800_640, max_slots=100)
    assert r["slots"] == 100, r


def test_plan_expert_slots_reserve_reduces_capacity():
    from tinyinfer.kv_tiers import plan_expert_slots

    full = plan_expert_slots(devices=[{"name": "cuda:0", "free_bytes": 2_000_000_000}],
                             expert_bytes=18_800_640)
    less = plan_expert_slots(devices=[{"name": "cuda:0", "free_bytes": 2_000_000_000}],
                             expert_bytes=18_800_640, reserve_bytes=1_000_000_000)
    assert less["slots"] < full["slots"], (full, less)


def test_plan_expert_slots_no_devices():
    from tinyinfer.kv_tiers import plan_expert_slots

    r = plan_expert_slots(devices=[], expert_bytes=18_800_640)
    assert r["slots"] == 0
    assert r["devices"] == 0


def test_slots_needed_all():
    """40 层 x 384 专家 = 15360 —— 全装下才谈得上零搬运。"""
    from tinyinfer.kv_tiers import slots_needed_all

    assert slots_needed_all(40, 384) == 15360
    assert slots_needed_all(40, 6) == 240
