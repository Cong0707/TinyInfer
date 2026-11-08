"""压缩层的拼批一致性：目前**无法用单次 forward 测出来**。

为什么：压缩 KV 需要 compressor 跨步累积 pending —— 一次 forward 只喂一个
位置，`latent.size(1)` 是 0，写入分支根本不执行（实测 P=2 和 P=4 都是
`shared_attn.compress_kv is None`）。

要测它必须模拟「先走几步把 pending 攒满，再拼批」，那是另一套测试脚手架。
在那之前，这里显式记成 xfail，而不是假装通过或者假装失败。

## 已知的风险点（读代码看出来的，未实测）

写入侧已经改成逐行 scatter（`_per_row_start`），但阅读侧：

```python
compress_kv = self.rt.compress_kv[:, :compress_len]
```

`compress_len = end_pos // ratio_now` 是按**本批最大位置**算的，然后所有行都取
同一个 `[:, :compress_len]`。多行深度差很大时，浅的那行会读到深的那行的组 ——
那些组对它来说是「未来的位置」，语义上不该可见。

要修就得把 `compress_kv` 的可见范围也改成逐行（每行取 `[:, :own_len]`），
或者在 attention 里用 idxs 把越界组屏蔽掉。这是一个**真实的待办**，不是已解决项。
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
def comp_block():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    if "eng" not in _CACHE:
        from tinyinfer.engine import Engine
        from tinyinfer.tiers import OptLevel
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(str(MODEL), trust_remote_code=True)
        eng = Engine(str(MODEL), tier=OptLevel.AGGRESSIVE, device="cuda",
                     layers=[2], tokenizer=tok)
        _CACHE["eng"] = eng
    eng = _CACHE["eng"]
    yield eng, eng._blocks[2]


@pytest.mark.xfail(
    reason="a single forward never materialises a compressed group (the "
           "compressor accumulates pending across steps), so batching cannot "
           "be compared within one call; the read side is still per-batch, see "
           "the module docstring",
    strict=False,
)
def test_compressed_layer_batching(comp_block):
    """占位：等有了跨步脚手架之后，这里应该改成真正的一致性断言。"""
    eng, b = comp_block
    from tinyinfer.deepseek.compressor import shared_attn as _sa

    from tinyinfer.nn import rms_norm

    HC, D = eng.cfg.hc_mult, eng.cfg.dim
    torch.manual_seed(0)
    x1 = torch.randn(1, 1, HC, D, dtype=torch.bfloat16, device="cuda") * 0.02
    pm = torch.zeros(1, 1, HC, dtype=torch.float32, device="cuda")
    pm[:, :, 0] = 1.0
    h1 = rms_norm(b.hc_pre(x1, pm), b.attn_norm, eng.cfg.rms_norm_eps)

    for sub in (b.attn, getattr(b.attn, "compressor", None),
                getattr(b.attn, "indexer", None)):
        if sub is not None and hasattr(sub, "reset"):
            sub.reset()
    _sa.compress_kv = None

    P = 4
    b.attn.forward(h1, P, positions=torch.tensor([P], device="cuda"))
    assert _sa.compress_kv is not None, (
        "a single forward did not write a compressed group, so this comparison "
        "is meaningless -- needs a multi-step harness"
    )
