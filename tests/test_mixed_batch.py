"""混批（一行 prefill、一行 decode）在什么前提下成立。

sched.ServingEngine 现在的策略是「有 fresh 序列时只推进它自己」，理由是混批
会抛错。上一轮修掉了 attention 里按行算窗口索引的那个错误，但混批还有一条
真实的语义限制：**同一 batch 里各行的列数必须相同**（拼成一个矩形张量），
而 prefill 行有 prompt_len 列、decode 行只有 1 列。

所以正确做法是在拼批时按最宽的行补 pad，并让每行各自的 positions/窗口索引
按自己的有效长度算 —— 这一组测试锁住「等宽两行」这个前提。
"""

import pytest

torch = pytest.importorskip("torch")

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from conftest import model_dir  # noqa: E402

MODEL = model_dir()
_CACHE = {}


@pytest.fixture()
def engine_pair():
    eng = _CACHE.get("eng")
    if eng is None:
        if not torch.cuda.is_available():
            pytest.skip("needs CUDA")
        from tinyinfer.engine import Engine
        from tinyinfer.tiers import OptLevel
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(str(MODEL), trust_remote_code=True)
        eng = Engine(str(MODEL), tier=OptLevel.AGGRESSIVE, device="cuda",
                     layers=list(range(2)), tokenizer=tok)
        _CACHE["eng"] = eng
        _CACHE["tok"] = tok
    yield _CACHE["eng"], _CACHE["tok"]


def test_two_rows_same_width_and_different_positions(engine_pair):
    """两行等宽、但各自的起点不同 —— 这才是混批的常见形态。

    一行刚开始（pos=0）、一行已经走了几步（pos=4）。窗口索引必须各行按自己的
    位置算，否则第二行会读到第一行的历史槽位。
    """
    eng, tok = engine_pair
    eng.reset()
    pad = getattr(eng, "pad_id", 0) or 0
    w = 2
    ids = torch.full((2, w), pad, dtype=torch.long, device="cuda")
    ids[0, 0] = 100
    ids[1, 0] = 200
    pos = torch.tensor([0, 4], dtype=torch.long, device="cuda")

    # 对照：同一行单独跑，和拼批后应有相同的数值
    eng.reset()
    one = eng.forward(ids[:1], 0, positions=pos[:1])
    eng.reset()
    two = eng.forward(ids, 0, positions=pos)

    assert two.shape[0] == 2, two.shape
    assert torch.isfinite(two).all()
    # 行 0 在两种情况下输入完全相同，输出必须一致（拼批不能改变任何一行的结果）
    d = float((two[0].float() - one[0].float()).abs().max())
    assert d < 1e-3, f"batched row differs from the single-row run by {d}"
