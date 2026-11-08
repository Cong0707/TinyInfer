"""拼批的数值一致性：单行与 batch=2 必须逐值相同。

关键约定（踩过坑）：start_pos 必须与本批**最小位置**一致，positions 是每行
每个位置的绝对位置。两者不一致时单行与拼批走的分支不同（window 索引一条按
start_pos 算、一条按 positions 每行算），看起来像「batch 支持有问题」，其实
是调用方传错了参数。

实测（layers=[0]，随机 hidden，start_pos=1、positions=[1,1]）：
  attention row0 vs single = 0.000000
  MoE/ffn  row0 vs single = 0.000015（浮点噪声）
"""

import pytest

torch = pytest.importorskip("torch")

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from conftest import model_dir  # noqa: E402

pytestmark = pytest.mark.slow   # 要加载真 checkpoint 并独占引擎锁

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


def test_attention_batching_matches_single_row(block_pair):
    """一行和两行（两行数据相同）的输出必须逐值相同。"""
    eng, b = block_pair
    from tinyinfer.nn import rms_norm

    HC, D = eng.cfg.hc_mult, eng.cfg.dim
    torch.manual_seed(0)
    x1 = torch.randn(1, 1, HC, D, dtype=torch.bfloat16, device="cuda") * 0.02
    pm = torch.zeros(1, 1, HC, dtype=torch.float32, device="cuda")
    pm[:, :, 0] = 1.0
    h1 = rms_norm(b.hc_pre(x1, pm), b.attn_norm, eng.cfg.rms_norm_eps)

    P = 1
    _fresh(b)
    one = b.attn.forward(h1, P, positions=torch.tensor([P], device="cuda")).float()
    _fresh(b)
    two = b.attn.forward(torch.cat([h1, h1], 0), P,
                         positions=torch.tensor([P, P], device="cuda")).float()

    d = float((two[0:1] - one).abs().max())
    assert d == 0.0, f"batched row differs from the single-row run by {d}"
    assert float((two[0] - two[1]).abs().max()) == 0.0, "identical rows must match"


def test_ffn_batching_matches_single_row(block_pair):
    """MoE 拼批不改变任何一行的结果（只有浮点噪声）。"""
    eng, b = block_pair
    from tinyinfer.nn import rms_norm

    HC, D = eng.cfg.hc_mult, eng.cfg.dim
    torch.manual_seed(0)
    # 两行必须是同一份输入：MoE 的路由由数据决定，两行数据不同时各自路由到
    # 不同专家，拼批和单行本来就会有差异（那是模型的语义，不是 bug）。
    x1 = torch.randn(1, 1, HC, D, dtype=torch.bfloat16, device="cuda") * 0.02
    x2 = torch.cat([x1, x1], 0)

    def run(x, n):
        pm = torch.zeros(n, 1, HC, dtype=torch.float32, device="cuda")
        pm[:, :, 0] = 1.0
        h = rms_norm(b.hc_pre(x, pm), b.ffn_norm, eng.cfg.rms_norm_eps)
        return b.ffn.forward(h).float()

    one = run(x1, 1)
    two = run(x2, 2)
    assert torch.isfinite(two).all(), "batched MoE produced non-finite values"
    d = float((two[0:1] - one).abs().max())
    assert d < 1e-2, f"MoE batching changed row 0 by {d}"
