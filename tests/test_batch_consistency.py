"""拼批的数值一致性：哪一部分支持、哪一部分不支持。

实测（layers=[0]，随机 hidden，pos=[1,2]）：

    attn  row0 maxdiff = 4.70      ← 完全不对
    ffn   row0 maxdiff = 0.000015  ← 只是浮点噪声

所以 MoE 侧拼批是正确的（本轮修掉了它分批时的三个 bug），卡点在 attention。

原因很直接：window KV 是**一份**环形缓冲（每层一个），位置 p 写进 p % window。
两个位置不同的序列往同一份 buffer 里写，第二个会把第一个的历史覆盖掉。
压缩 KV 也一样（本轮改成逐行写之后至少不互相踩了，但阅读侧仍按单一起点取）。

结论：
  - batch>1 在 MoE/线性层上是正确的；
  - attention 要支持真并发，必须让每个序列有自己的 window buffer
    （按 slot 索引，而不是每层一份）；
  - 在那之前，sched 的批宽必须压到「所有在飞序列位置相同」的程度，
    或者干脆 1。

这一组测试把「MoE 拼批正确」这个已经成立的事实锁住，避免以后又改坏。
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
def block_pair():
    """一个单层 block，用来分别量 attn 和 ffn 的拼批一致性。"""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    b = _CACHE.get("block")
    if b is None:
        from tinyinfer.engine import Engine
        from tinyinfer.tiers import OptLevel
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(str(MODEL), trust_remote_code=True)
        eng = Engine(str(MODEL), tier=OptLevel.AGGRESSIVE, device="cuda",
                     layers=[0], tokenizer=tok)
        _CACHE["eng"] = eng
        _CACHE["block"] = eng._blocks[0]
        b = _CACHE["block"]
    yield _CACHE["eng"], b


def test_ffn_batching_is_numerically_identical(block_pair):
    """MoE 拼批不改变任何一行的结果（只有浮点噪声）。

    这是本轮修的三件事的可执行形式：分批按行切、按 dim 0 拼、还原 [b,s,hc,d]。
    """
    eng, b = block_pair
    from tinyinfer.nn import rms_norm

    HC, D = eng.cfg.hc_mult, eng.cfg.dim
    torch.manual_seed(0)
    x = torch.randn(2, 1, HC, D, dtype=torch.bfloat16, device="cuda") * 0.02

    def run(n):
        pm = torch.zeros(n, 1, HC, dtype=torch.float32, device="cuda")
        pm[:, :, 0] = 1.0
        h = b.hc_pre(x[:n], pm)
        h = rms_norm(h, b.ffn_norm, eng.cfg.rms_norm_eps)
        return b.ffn.forward(h).float()

    one = run(1)
    two = run(2)
    d = float((two[0] - one[0]).abs().max())
    assert d < 1e-3, f"MoE batching changed row 0 by {d}"


def test_attention_batching_is_known_broken(block_pair):
    """attention 拼批目前**不正确**：一份 window buffer 装不下两个历史。

    这条测试不是在断言「应该坏」，是在记录**已知边界**：它一旦通过，说明
    有人给每个序列做了独立的 window buffer，那时应该把它改成一致性断言。
    """
    eng, b = block_pair
    from tinyinfer.nn import rms_norm

    HC, D = eng.cfg.hc_mult, eng.cfg.dim
    torch.manual_seed(0)
    x = torch.randn(2, 1, HC, D, dtype=torch.bfloat16, device="cuda") * 0.02
    pos = torch.tensor([1, 2], dtype=torch.long, device="cuda")

    def reset():
        for sub in (b.attn, getattr(b.attn, "compressor", None),
                    getattr(b.attn, "indexer", None)):
            if sub is not None and hasattr(sub, "reset"):
                sub.reset()

    def run(n):
        reset()
        pm = torch.zeros(n, 1, HC, dtype=torch.float32, device="cuda")
        pm[:, :, 0] = 1.0
        h = b.hc_pre(x[:n], pm)
        h = rms_norm(h, b.attn_norm, eng.cfg.rms_norm_eps)
        return b.attn.forward(h, 0, positions=pos[:n]).float()

    one = run(1)
    two = run(2)
    d = float((two[0] - one[0]).abs().max())
    assert d > 1e-2, (
        f"attention batching now agrees (diff {d}) -- that means per-sequence "
        "window buffers landed; flip this test to an equality assertion")
