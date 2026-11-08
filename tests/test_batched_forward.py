"""batch=2 前向：并发能不能真跑起来的分界线。

sched.ServingEngine 的注释里写着「batch 维从 1 变成 2 时会抛
output with shape [1, T, 512] doesn't match [2, T, 512]，实测 10 次请求
2-3 次 500」，所以它把 max_batch 压到 1。那条限制就是并发上限。

这一组测试直接跑真模型的两条序列拼批，把每一个形状假设都暴露出来。
"""

import pytest

# 这一组要加载真 checkpoint 并独占引擎锁（一台机器只能有一个引擎），
# 所以按 slow 归类：默认门禁跳过，用 pytest -m slow 单跑。
pytestmark = pytest.mark.slow

torch = pytest.importorskip("torch")

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from conftest import model_dir  # noqa: E402

MODEL = model_dir()


def _engine(n_layers=2):
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from tinyinfer.engine import Engine
    from tinyinfer.tiers import OptLevel
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(MODEL), trust_remote_code=True)
    eng = Engine(str(MODEL), tier=OptLevel.AGGRESSIVE, device="cuda",
                 layers=list(range(n_layers)), tokenizer=tok)
    return eng, tok


_CACHE = {}


@pytest.fixture()
def engine_pair():
    """整模块共用一个引擎。

    两个原因：引擎有机器级锁（并存会 EngineLockBusy），8GB 卡也放不下两份
    模型（第二个在 setup 阶段就 OOM）。每个测试开头自己 reset() 就够了。
    """
    eng = _CACHE.get("eng")
    if eng is None:
        eng, tok = _engine()
        _CACHE["eng"] = eng
        _CACHE["tok"] = tok
    yield _CACHE["eng"], _CACHE["tok"]


def test_two_sequences_in_one_forward(engine_pair):
    """两条序列拼成一个 batch 走一次 forward，形状必须全程一致。"""
    eng, tok = engine_pair
    eng.reset()
    a = tok.encode("你好", add_special_tokens=False)
    b = tok.encode("再见", add_special_tokens=False)
    n = min(len(a), len(b))
    ids = torch.tensor([a[:n], b[:n]], dtype=torch.long, device="cuda")
    pos = torch.arange(n, device="cuda").unsqueeze(0).expand(2, -1)
    logits = eng.forward(ids, 0, positions=pos)
    assert logits.shape[0] == 2, logits.shape
    assert logits.shape[1] == n, logits.shape
    assert torch.isfinite(logits).all()


def test_batch_one_then_two(engine_pair):
    """先单条、再拼批：缓存过的中间张量不能停在 batch=1 的形状。"""
    eng, tok = engine_pair
    eng.reset()
    a = tok.encode("你好", add_special_tokens=False)
    one = torch.tensor([a], dtype=torch.long, device="cuda")
    pos1 = torch.arange(len(a), device="cuda").unsqueeze(0)
    eng.forward(one, 0, positions=pos1)

    b = tok.encode("再见", add_special_tokens=False)
    n = min(len(a), len(b))
    ids = torch.tensor([a[:n], b[:n]], dtype=torch.long, device="cuda")
    pos2 = torch.arange(n, device="cuda").unsqueeze(0).expand(2, -1)
    logits = eng.forward(ids, 0, positions=pos2)
    assert logits.shape[0] == 2
    assert torch.isfinite(logits).all()
