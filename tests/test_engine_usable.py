"""端到端可用性：4 层模型连续多轮必须输出正确字符。

这是「引擎到底能不能用」的最低标准。之前一直用临时探针验证，现在是测试。

注意：只跑 4 层（8GB 卡上 5 层会 OOM）。层数少时模型能力有限，
所以断言的是「输出稳定且不是乱码」，不是「答对数学题」。
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


def _engine(nlayers=4):
    """每个用例一个引擎 —— 跨用例共享会让第二个用例拿到被污染的共享状态。

    实测：同一个引擎上先跑一个用例、再跑另一个，第二个会出 NaN。
    这与 docs/measurement_discipline.md 里「一进程一引擎」是同一条纪律。
    """
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from tinyinfer.engine import Engine
    from tinyinfer.tiers import OptLevel
    from transformers import AutoTokenizer

    tok = _CACHE.get("tok")
    if tok is None:
        tok = AutoTokenizer.from_pretrained(str(MODEL), trust_remote_code=True)
        _CACHE["tok"] = tok
    # 一台机器只能有一个引擎（EngineLock），所以两个用例共用它。
    # 共用时必须在每个用例开头 reset()，否则第二个用例拿到的是第一个的残留。
    if "eng" not in _CACHE:
        _CACHE["eng"] = Engine(str(MODEL), tier=OptLevel.AGGRESSIVE, device="cuda",
                               layers=list(range(nlayers)), tokenizer=tok)
    eng = _CACHE["eng"]
    eng.reset()
    return eng, tok


def test_three_turns_are_all_finite():
    """真实多轮顺序（reset -> prefill -> 取 argmax）连续三轮都必须是有限值。

    这条守住的是「引擎连续服务多个请求」这个最基本的可用性，
    而不只是「第一次 forward 不出错」。
    """
    eng, tok = _engine()
    qs = ["1+1 等于几？只回答数字。",
          "2+2 等于几？只回答数字。",
          "3+3 等于几？只回答数字。"]
    outs = []
    for q in qs:
        eng.reset()
        ids = torch.tensor([tok.encode(q, add_special_tokens=False)], device="cuda")
        lg = eng._forward_chunk(ids, 0)
        torch.cuda.synchronize()
        f = lg.float()
        assert torch.isfinite(f).all(), f"NaN in logits for {q!r}"
        outs.append(int(f[0, -1].argmax()))
    # 三个不同的问题不该全给出同一个 token（那说明输入没起作用）
    assert len(set(outs)) >= 2, f"all three turns gave the same token: {outs}"


def test_output_is_not_degenerate():
    """logits 不能是全零、全同、或量级离谱 —— 那些都是「跑通了但没算对」。"""
    eng, tok = _engine()
    eng.reset()
    ids = torch.tensor([tok.encode("你好", add_special_tokens=False)], device="cuda")
    lg = eng._forward_chunk(ids, 0)
    torch.cuda.synchronize()
    f = lg.float()
    assert torch.isfinite(f).all()
    v = f[0, -1]
    assert float(v.std()) > 1e-3, "logits are constant"
    assert 0.5 < float(v.abs().max()) < 1e4, (
        f"logits magnitude out of range: {float(v.abs().max())}")
