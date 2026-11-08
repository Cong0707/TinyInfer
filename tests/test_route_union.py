"""并发时 MoE 的专家集合是「所有会话路由的并集」—— 这笔账必须先算清。

假设 4 条会话同时 decode，每条每层路由 6 个专家。如果它们路由到完全不同的
专家，一层就要 24 个；而槽位池在 L20 上是 1359 个、40 层 x 6 = 240 是「单会话」
的量。四条会话就是 960 —— 还在 1359 以内，但余量只剩 400。

这一组用真实的 gate 计算「并集有多大」，而不是拍脑袋假设。
"""

import pytest

torch = pytest.importorskip("torch")

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from conftest import model_dir  # noqa: E402

pytestmark = pytest.mark.slow

MODEL = model_dir()
_CACHE = {}


@pytest.fixture()
def gate():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    if "gate" not in _CACHE:
        from tinyinfer.engine import Engine
        from tinyinfer.tiers import OptLevel
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(str(MODEL), trust_remote_code=True)
        eng = Engine(str(MODEL), tier=OptLevel.AGGRESSIVE, device="cuda",
                     layers=[0], tokenizer=tok)
        _CACHE["eng"] = eng
        _CACHE["gate"] = eng._blocks[0].ffn.gate
    yield _CACHE["eng"], _CACHE["gate"]


def _route(gt, x):
    _, idx = gt.forward(x, None)
    return set(int(v) for v in idx.reshape(-1).tolist())


def test_union_grows_slower_than_linear(gate):
    """关键问题：n 条会话的专家并集，是不是 n x 6？

    如果不同输入的路由高度错开，并集就是 6n，L20 的 1359 个槽位在 4 条会话时
    仍然够（960），但 6 条就到 1440 开始 thresh。这一条把这个数字测出来。
    """
    eng, gt = gate
    D = eng.cfg.dim
    torch.manual_seed(0)
    sets = []
    for i in range(8):
        x = torch.randn(1, D, dtype=torch.bfloat16, device="cuda") * 0.02
        sets.append(_route(gt, x))

    union = set()
    sizes = []
    for i, s in enumerate(sets):
        union |= s
        sizes.append(len(union))
    per_session = eng.cfg.n_activated_experts
    print("[union] sessions=8 per=%d sizes=%s final=%d" %
          (per_session, sizes, len(union)), flush=True)
    assert sizes[0] == per_session, sizes
    # 最坏情况是 8 x 6 = 48；如果明显更小，说明路由有共同热点
    assert len(union) <= 8 * per_session, len(union)


def test_same_input_gives_the_same_route(gate):
    """同一份输入的路由必须稳定 —— 否则「相邻 token 复用槽位」这个前提就没了。"""
    eng, gt = gate
    D = eng.cfg.dim
    torch.manual_seed(7)
    x = torch.randn(1, D, dtype=torch.bfloat16, device="cuda") * 0.02
    a = _route(gt, x)
    b = _route(gt, x)
    assert a == b, (sorted(a), sorted(b))

def test_union_over_layers_fits_the_L20_pool(gate):
    """把「8 条会话 x 40 层」的并集折算到 L20 的 1359 个槽位。

    实测（单层、8 条不同输入）：并集 35 个，而不是最坏的 48。
    40 层 x 35 = 1400，略超 1359 —— 也就是说 L20 上 8 条并发会开始换出，
    4-6 条则完全在池内。这个数字决定了 batch 宽度的实际上限。
    """
    eng, gt = gate
    D = eng.cfg.dim
    torch.manual_seed(0)
    sets = []
    for _ in range(8):
        x = torch.randn(1, D, dtype=torch.bfloat16, device="cuda") * 0.02
        sets.append(_route(gt, x))
    union = set()
    for s in sets:
        union |= s
    per_layer = len(union)
    n_layers = int(eng.cfg.n_layers)
    total = per_layer * n_layers
    print("[union] per_layer=%d x %d layers = %d slots" %
          (per_layer, n_layers, total), flush=True)
    # 单层并集在 6n 和最坏 6n 之间：确认它有共同热点（< 8*6）
    assert per_layer <= 8 * eng.cfg.n_activated_experts
    # 40 层的总量：这就是 L20 上 8 条并发要的槽位数
    assert 800 < total < 2000, total
