"""positions 两种约定的唯一换算入口。

这块出过三次静默错误（不报错、数值偏），每次都是不同调用点各按各的理解用：

  1. start_pos 与 positions 不一致（单行按 start_pos 切片、按行按 positions 算）
  2. attention 只支持 [b, s]，decode 传的 [b] 走进错误分支
  3. compressed 把「每行最后一个位置」当「每行首列」传给 window_idxs_per_row

现在统一走 nn.normalize_positions，这组测试把两种约定的换算钉死。
"""

import pytest

torch = pytest.importorskip("torch")

from tinyinfer.nn import normalize_positions  # noqa: E402


def test_none_passes_through():
    assert normalize_positions(None, 1, 1) == (None, None, None)


def test_b_s_kept_as_is():
    p = torch.tensor([[10, 11, 12], [20, 21, 22]])
    pos2d, first, last = normalize_positions(p, 2, 3)
    assert pos2d.shape == (2, 3)
    assert first.tolist() == [10, 20]
    assert last.tolist() == [12, 22]


def test_b_means_last_position():
    """[b] 的语义是「每行最后一个位置」，s=1 时首=末。"""
    p = torch.tensor([7, 9])
    pos2d, first, last = normalize_positions(p, 2, 1)
    assert pos2d.shape == (2, 1)
    assert first.tolist() == [7, 9]
    assert last.tolist() == [7, 9]


def test_b_with_longer_seqexpands_backwards():
    """s=3、每行最后位置是 7 -> 这一段是 [5, 6, 7]。"""
    p = torch.tensor([7])
    pos2d, first, last = normalize_positions(p, 1, 3)
    assert pos2d.tolist() == [[5, 6, 7]]
    assert first.tolist() == [5]
    assert last.tolist() == [7]


def test_two_rows_different_depths():
    p = torch.tensor([3, 9])
    pos2d, first, last = normalize_positions(p, 2, 2)
    assert pos2d.tolist() == [[2, 3], [8, 9]]
    assert first.tolist() == [2, 8]
    assert last.tolist() == [3, 9]
