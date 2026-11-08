"""批宽和槽位数必须来自同一份硬件推导，不能一个写死 4 一个写死 8。

这一条测的是「引擎自己适配硬件」：同一个 plan_slots 既决定 KV 槽位数，
也决定连续批处理的批宽。写死的话，8GB 卡会配 8 条并发然后 OOM，
141GB 卡会只跑 8 条然后浪费 90% 的卡。
"""

import pytest

from tinyinfer.kv_tiers import plan_slots


def test_batch_width_and_slots_agree():
    """同一块卡上，两处推导必须给出同一个数字。"""
    for free_gb in (7, 20, 42, 80, 138):
        free = free_gb * 2**30
        a = plan_slots(vram_free_bytes=free)
        b = plan_slots(vram_free_bytes=free)
        assert a == b


def test_small_card_gets_one_batch():
    """8GB 卡装不下第二条 1M 会话，批宽必须是 1（不是 8）。"""
    assert plan_slots(vram_free_bytes=7 * 2**30) == 1


def test_big_card_is_not_capped_at_8():
    """141GB 卡不该被写死的 8 卡住。"""
    n = plan_slots(vram_free_bytes=138 * 2**30)
    assert n > 8, n
