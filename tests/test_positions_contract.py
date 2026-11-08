"""decode 时 positions 的约定：[b]，每行最后一个位置。

`engine._forward_chunk` 的注释写着 positions 是「每行最后一个 token 的位置」，
而 attention 里我还按 [b, s] 处理过 —— 两种约定必须都支持，否则 decode 的
多行 batch 会静默用错 RoPE 角度（不报错，数值偏）。

这一组锁住 [b] 语义：两行在同一位置，输出必须和单行逐值相同。
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
def block_pair():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    if "block" not in _CACHE:
        from tinyinfer.engine import Engine
        from tinyinfer.tiers import OptLevel
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(str(MODEL), trust_remote_code=True)
        eng = Engine(str(MODEL), tier=OptLevel.AGGRESSIVE, device="cuda",
                     layers=[0], tokenizer=tok)
        _CACHE["eng"] = eng
        _CACHE["block"] = eng._blocks[0]
    yield _CACHE["eng"], _CACHE["block"]


def _fresh(b):
    b.attn.reset()
    b.attn._kv_buf = None
    b.attn._decode_idxs = None


def test_decode_positions_b_semantics(block_pair):
    """positions 传 [b]（每行最后一个位置）时，两行结果仍要一致。"""
    eng, b = block_pair
    from tinyinfer.nn import rms_norm

    HC, D = eng.cfg.hc_mult, eng.cfg.dim
    torch.manual_seed(0)
    x1 = torch.randn(1, 1, HC, D, dtype=torch.bfloat16, device="cuda") * 0.02
    pm = torch.zeros(1, 1, HC, dtype=torch.float32, device="cuda")
    pm[:, :, 0] = 1.0
    h1 = rms_norm(b.hc_pre(x1, pm), b.attn_norm, eng.cfg.rms_norm_eps)

    P = 5
    # 单行：[1] 形状
    _fresh(b)
    one = b.attn.forward(h1, P, positions=torch.tensor([P], device="cuda")).float()
    # 双行：[2] 形状（每行一个位置）
    _fresh(b)
    two = b.attn.forward(torch.cat([h1, h1], 0), P,
                         positions=torch.tensor([P, P], device="cuda")).float()

    assert torch.isfinite(two).all()
    d = float((two[0:1] - one).abs().max())
    assert d == 0.0, f"[b] positions changed row 0 by {d}"
