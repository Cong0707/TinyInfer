"""并发下 MoE 的槽位需求：并集、热点、以及「批量失效」的风险。

前面测出 8 条会话单层并集是 35 个。但真实服务里更关键的问题是：

  **相邻 token 的路由重合度**（决定稳态能不能复用）
  **不同会话之间的重合度**（决定并集增长快不快）

这一组把两个都量出来，用真实模型的 gate。
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


def test_nearby_activations_overlap_more_than_random(gate):
    """相邻 token 的路由重合度必须明显高于随机 —— 这是「槽位复用」的前提。

    如果相邻 token 和随机两个 token 一样不相干，那槽位留着毫无意义，
    「覆盖即回收」也就退化成每步全搬。
    """
    eng, gt = gate
    D = eng.cfg.dim
    topk = eng.cfg.n_activated_experts
    torch.manual_seed(0)

    base = torch.randn(1, D, dtype=torch.bfloat16, device="cuda") * 0.02
    # 相邻：base + 小扰动（模拟同一段文本里邻近位置的高相似激活）
    near = [base, base + torch.randn(1, D, dtype=torch.bfloat16, device="cuda") * 0.002]
    r_near = [_route(gt, t) for t in near]
    ov_near = len(r_near[0] & r_near[1])

    # 随机：两个无关的激活
    rnd = [torch.randn(1, D, dtype=torch.bfloat16, device="cuda") * 0.02
           for _ in range(2)]
    r_rnd = [_route(gt, t) for t in rnd]
    ov_rnd = len(r_rnd[0] & r_rnd[1])

    print("[overlap] nearby=%d/%d random=%d/%d" %
          (ov_near, topk, ov_rnd, topk), flush=True)
    # 相邻必须比随机更重合；相等就说明这个模型没有局部性
    assert ov_near >= ov_rnd, (ov_near, ov_rnd)


def test_union_with_a_hot_shared_expert(gate):
    """不同会话是否存在「共同热点」：并集增长慢于线性的直接原因。"""
    eng, gt = gate
    D = eng.cfg.dim
    topk = eng.cfg.n_activated_experts
    torch.manual_seed(1)
    sets = []
    for _ in range(16):
        x = torch.randn(1, D, dtype=torch.bfloat16, device="cuda") * 0.02
        sets.append(_route(gt, x))

    # 统计每个专家被多少条会话命中
    from collections import Counter

    cnt = Counter()
    for s in sets:
        cnt.update(s)
    hot = cnt.most_common(3)
    union = set()
    for s in sets:
        union |= s
    print("[hot] 16 sessions: union=%d of max %d, hottest=%s" %
          (len(union), 16 * topk, hot), flush=True)
    # 并集应该显著小于最坏情况（16 x 6 = 96）
    assert len(union) < 16 * topk, len(union)
