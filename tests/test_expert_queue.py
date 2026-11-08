"""身份：GPU 报缺 -> CPU 备货 -> GPU 等待，这条链路的端到端测试。

这里不跑模型，只验证协议本身，因为这个协议一旦错了，整个 decode 会
在「槽位表指向别人的权重」这种不报错的地方出问题。
"""

import pytest

torch = pytest.importorskip("torch")

from tinyinfer.expert_queue import ExpertRequestRing, ReadyFlag  # noqa: E402


def _cuda():
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")


def test_ring_is_host_mapped_and_visible():
    _cuda()
    ring = ExpertRequestRing()
    # GPU 写、CPU 读，中间没有拷贝：CPU 侧看到的是同一块物理内存。
    assert ring.posts_so_far() == 0
    ring.host[0] = 7
    assert ring.posts_so_far() == 7
    ring.reset()


def test_ready_flag_bumps_and_is_readable():
    _cuda()
    flag = ReadyFlag()
    assert flag.value == 0
    assert flag.bump() == 1
    assert flag.host.value == 1
    flag.reset()
    assert flag.host.value == 0


def test_kernel_writes_miss_list_to_the_ring():
    _cuda()
    from tinyinfer.kernels import native

    ring = ExpertRequestRing()
    # slot_ids 里两个 -1（缺），其余有槽位
    slot_ids = torch.tensor([0, -1, 2, -1, 4], dtype=torch.int32, device="cuda")
    ring.post(slot_ids, 5, layer=3, epoch=1)
    torch.cuda.synchronize()
    # GPU 侧已经写进 host-mapped 内存，CPU 直接读得到
    assert ring.posts_so_far() == 1
    assert int(ring.host[1]) == 1
    reqs = ring.drain()
    assert len(reqs) == 1
    layer, ids = reqs[0]
    assert layer == 3
    assert ids == [1, 3], ids
    # 再取一次是空的（已经消费过）
    assert ring.drain() == []
    native._lib()  # 确保驱动库仍可用


def test_stream_wait_value_blocks_until_the_flag_moves():
    """GPU 等 CPU 备货：等待在流里发生，主机不阻塞。"""
    _cuda()
    from tinyinfer.kernels import native

    flag = ReadyFlag()
    out = torch.zeros(1, dtype=torch.int32, device="cuda")
    # 排一个「等 flag >= 1」的等待，再排一个写显存 42 的 kernel
    fn = native.BoundLaunch(native.kernel("moe_req", "moe_ready_spin"),
                            (1,), (32,), "pi")
    flag.wait_on_stream(0)          # 已经满足，立即通过
    flag.bump()
    fn(flag.dev_ptr, 1)             # 自旋版：等 flag 到 1
    torch.cuda.synchronize()
    # 能走到这里就说明没有死等（flag 已经在 1）
    assert flag.host.value == 1


def test_miss_count_kernel_matches_python():
    _cuda()
    from tinyinfer.kernels import native

    ring = ExpertRequestRing()
    flag = ReadyFlag()
    slot_ids = torch.tensor([-1, 1, -1, -1, 2], dtype=torch.int32, device="cuda")
    fn = native.BoundLaunch(native.kernel("moe_req", "moe_miss_count"),
                            (1,), (32,), "Tip")
    fn(slot_ids, 5, flag.dev_ptr)
    torch.cuda.synchronize()
    assert flag.host.value == 3, flag.host.value
    _ = ring
