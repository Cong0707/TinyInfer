import pytest

torch = pytest.importorskip("torch")


def _cuda():
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")


def test_gpu_side_wait_replaces_the_host_sync():
    """GPU 自己判断缺谁、发布、并在流里等 CPU 备好 —— 主机侧不参与。"""
    _cuda()
    from tinyinfer.expert_queue import ExpertRequestRing

    ring = ExpertRequestRing()
    slot_ids = torch.tensor([0, -1, 2, -1, 4], dtype=torch.int32, device="cuda")
    ring.wait_ready(slot_ids, 5, layer=7, epoch=1)

    # GPU 已经发布（host-mapped，CPU 立刻看得到），然后卡在等就绪上。
    import time
    deadline = time.time() + 3.0
    while ring.posts_so_far() < 1 and time.time() < deadline:
        time.sleep(0.001)
    assert ring.posts_so_far() >= 1, "the GPU never published its request"
    reqs = ring.drain()
    assert reqs and reqs[0][0] == 7
    assert reqs[0][1] == [1, 3], reqs

    # 备好了：放开 GPU
    ring.set_ready(1)
    torch.cuda.synchronize()          # 必须能返回（不是死等）
    assert ring.host[2] == 1


def test_gpu_side_wait_is_a_noop_when_nothing_is_missing():
    """全在显存里时不该发布、也不该等。"""
    _cuda()
    from tinyinfer.expert_queue import ExpertRequestRing

    ring = ExpertRequestRing()
    slot_ids = torch.tensor([0, 1, 2], dtype=torch.int32, device="cuda")
    ring.wait_ready(slot_ids, 3, layer=0, epoch=1)
    torch.cuda.synchronize()          # 立即返回：没有缺失就没有等待
    assert ring.posts_so_far() == 0, "must not publish when nothing is missing"
