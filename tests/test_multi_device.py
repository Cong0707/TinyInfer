"""多卡 / EP：槽位分片到多张卡后，DMA 和等待必须打到正确的卡。

这一组测试锁住的是「spread the slot arenas over several cards」之后，
**消费侧**也跟着按卡分派。此前 slot_device() 只有测试在用，生产代码一条
都没走它 —— 多卡上把目标在 cuda:1 的 DMA 排进 cuda:0 的 stream，驱动会按
错误卡的地址空间解析目标指针。
"""

import pytest

torch = pytest.importorskip("torch")

from tinyinfer.loading import VramSlotCache  # noqa: E402

BLOB = 1024


def _two_cards():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    if torch.cuda.device_count() < 2:
        pytest.skip("needs two visible CUDA devices")


def test_each_slot_reports_its_card():
    """槽位分布到两张卡上，每个槽位都能说出自己在哪张卡。"""
    _two_cards()
    cache = VramSlotCache(8, BLOB, device="cuda:0", dtype=torch.uint8,
                          pin_staging=True, slot_arena=True)
    cache.set_devices(["cuda:0", "cuda:1"], [1, 1])
    cache._ensure()
    devs = {cache.slot_device(i) for i in range(cache.slots)}
    assert devs == {0, 1}, f"slots were not spread: {devs}"


def test_slot_stream_follows_the_slot_device():
    """每块槽位的 DMA stream 必须是它所在卡上的那一条。"""
    _two_cards()
    cache = VramSlotCache(8, BLOB, device="cuda:0", dtype=torch.uint8,
                          pin_staging=True, slot_arena=True)
    cache.set_devices(["cuda:0", "cuda:1"], [1, 1])
    cache._ensure()
    by_dev = {}
    for i in range(cache.slots):
        by_dev.setdefault(cache.slot_device(i), []).append(i)
    for dev, slots in by_dev.items():
        for i in slots:
            s = cache.slot_stream(i)
            assert s is not None
            assert s.device.index == dev, (i, dev, s.device)


def test_single_card_is_unchanged():
    """单卡上 slot_stream 就是原来那一条，路径不变。"""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    cache = VramSlotCache(4, BLOB, device="cuda:0", dtype=torch.uint8,
                          pin_staging=True, slot_arena=True)
    cache._ensure()
    assert cache.slot_device(0) == 0
    assert cache.slot_stream(0) is cache._copy_stream
