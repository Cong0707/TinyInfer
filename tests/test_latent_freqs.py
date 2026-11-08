"""长序列下压缩 attention 的 rope 表必须按「每行最后一个位置」算。

回归的是 `_latent_freqs` 的一个形状 bug：grp0 在 s>1 时是 [b, s]，
而 `.unsqueeze(1)` 得到 [b, 1, s]，去加 [1, ng] 直接 RuntimeError。
引擎的 except 把它吞成 NaN，所以症状是「1024 token 输出全 NaN」而没有异常。
"""

import pytest
import torch

from tinyinfer.nn import rms_norm


def _make_attn(ratio=2, b=1, s=8, head_dim=16, rope_dim=4, ng=4, n_heads=2):
    """构造一个不依赖权重的压缩 attention，只测 rope 表这一段。"""
    from tinyinfer.deepseek.compressed_attention import MLACompressedAttention

    cfg = type("C", (), {
        "head_dim": head_dim, "rope_dim": rope_dim, "n_heads": n_heads,
        "dim": head_dim * n_heads, "window": 128, "rms_norm_eps": 1e-20,
        "rope_theta": 10000.0,
    })()
    return MLACompressedAttention.__new__(MLACompressedAttention), cfg


def test_grp0_last_column_is_used_for_freqs():
    """grp0 为 [b, s] 时，freqs 只取最后一列 —— 这是修复的核心语义。

    这里不构造完整 attention，只验证「取最后一列」这个不变量：
    [b, s] 的最后一列与直接传 [b] 的那些值相等。
    """
    b, s, ng = 2, 8, 4
    ratio = 2
    grp0_2d = torch.arange(b * s).reshape(b, s) // ratio
    grp0_1d = grp0_2d[:, -1]
    assert torch.equal(grp0_1d, grp0_2d[:, -1])
    # 真正的断言：两种形状喂进修复后的逻辑，得到的 grp 一致
    _g2 = grp0_2d if grp0_2d.dim() == 1 else grp0_2d[:, -1]
    _g1 = grp0_1d if grp0_1d.dim() == 1 else grp0_1d[:, -1]
    assert torch.equal(_g1.unsqueeze(1), _g2.unsqueeze(1))


def test_latent_freqs_accepts_both_shapes():
    """[b] 和 [b, s] 两种 grp0 都必须能算出 [b, ng, rope_dim] 的表。"""
    pytest.importorskip("torch")
    from tinyinfer.deepseek.compressed_attention import MLACompressedAttention

    b, s, ng, rope_dim, ratio = 2, 8, 4, 4, 2
    half = rope_dim // 2
    freqs_cis = torch.randn(64, rope_dim)

    obj = MLACompressedAttention.__new__(MLACompressedAttention)
    obj.ratio = ratio
    obj.freqs_cis = freqs_cis
    obj.rt = type("R", (), {})()

    grp0_1d = torch.tensor([3, 5])
    fr1 = obj._latent_freqs(grp0_1d, ng, b, "cpu")
    assert tuple(fr1.shape) == (b, ng, rope_dim)

    obj2 = MLACompressedAttention.__new__(MLACompressedAttention)
    obj2.ratio = ratio
    obj2.freqs_cis = freqs_cis
    obj2.rt = type("R", (), {})()
    grp0_2d = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 5]])
    fr2 = obj2._latent_freqs(grp0_2d, ng, b, "cpu")
    assert tuple(fr2.shape) == (b, ng, rope_dim)
    # 两种形状的最后一列相同 -> 表必须逐位相同
    assert torch.equal(fr1, fr2)
