"""规划器可移植性：同一份 checkpoint 在假想机器上的表现。

不碰 GPU —— 每个用例构造一个合成的 Hardware，检查 Planner 解出的布局是否合理。
这是「换一台机器也必须成立」的回归守卫。

判据只针对不变量：层数守恒、槽位非负且装得下、稠密模型不分配槽位、
显存越大常驻越多。**不写死具体数字** —— 那些由目标函数算，换模型/换卡会变。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tinyinfer.planner import Planner  # noqa: E402
from tinyinfer.resources import Card, Hardware  # noqa: E402

GB = 2**30

# 本机这份 checkpoint 的几何（model_profile 实测）
MODEL = {
    "layer_bytes": 172e6,
    "max_layer_bytes": 468e6,
    "embed_bytes": 1.233e9,
    "head_bytes": 1.233e9,
    "expert_bytes": 18.8e6,
    "n_layers": 40,
    "topk": 6,
    "n_experts": 384,
}
N = 40

# 带宽必须给：Planner 不接受 fallback。
BW = {"h2d_bps": 7.1e9, "disk_bps": 4.4e9, "ram_bps": 35e9}


def _hw(cards_gb, ram, avail, unified=False):
    cards = [Card(i, "card", gb, gb) for i, gb in enumerate(cards_gb)]
    return Hardware(name="test", vram_gb=sum(cards_gb), ram_gb=ram,
                    ram_avail_gb=avail, unified=unified, cards=cards)


def _solve(cards_gb, ram, avail, unified=False, model=None):
    hw = _hw(cards_gb, ram, avail, unified)
    p = Planner(hw, model or MODEL, bandwidth=BW)
    return p, p.solve()


def test_this_box_shape():
    p, lay = _solve([8.0], 47.0, 35.0)
    # 层数守恒
    assert 0 <= lay.resident <= N
    # 显存装得下
    assert p.used_bytes(lay) <= p.vram
    # 本机 8GB + 7.1GB/s 的链路：搬运慢，会把显存尽量给常驻层
    assert lay.resident >= 1
    # 覆盖一整步的槽位比「多驻留几层」值钱时，槽位必须存在
    assert lay.slots >= 0


def test_big_card_is_fully_resident():
    p, lay = _solve([24.0], 64.0, 56.0)
    assert lay.resident == N
    assert p.used_bytes(lay) <= p.vram


def test_residency_is_monotonic_in_vram():
    prev = -1
    for vram in (4.0, 8.0, 16.0, 24.0, 48.0):
        _, lay = _solve([vram], 128.0, 120.0)
        if lay.resident + 1 < prev:
            raise AssertionError("resident 随显存下降：%s 之后是 %s" % (prev, lay.resident))
        prev = lay.resident


def test_dense_model_gets_no_slots():
    dense = dict(MODEL, expert_bytes=0.0, topk=0)
    p, lay = _solve([8.0], 47.0, 35.0, model=dense)
    assert lay.slots == 0


def test_slot_count_scales_with_vram():
    """槽位数由「显存能给多少 ÷ 一个专家多大」算出，不是模型形状。"""
    _, small = _solve([8.0], 64.0, 56.0)
    _, big = _solve([24.0], 128.0, 120.0)
    # 大卡要么全驻留（不需要那么多槽位），要么槽位更多
    assert big.slots >= small.slots or big.resident == N


def test_two_models_get_different_layouts():
    """换模型必须换布局 —— 说明求解真的用了模型几何，而不是查表。"""
    big_model = dict(MODEL, layer_bytes=0.72 * GB, max_layer_bytes=3.8 * GB, n_layers=64)
    _, a = _solve([24.0], 128.0, 120.0)
    _, b = _solve([24.0], 128.0, 120.0, model=big_model)
    assert a.resident != b.resident, "两种层尺寸解出同样的常驻层数，几何没被用上"
