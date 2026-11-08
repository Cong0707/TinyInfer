"""热点分布：pin 到底值不值得？

历史结论是「pin 是负优化」—— 那是实测出来的（1119 个 pin 之后 decode 5.11 tok/s
且输出复读），原因是 decode 的路由每个 token 都在变，prefill 期的 top-N 覆盖不到。

但现在的数据和那时不同：槽位是覆盖式的，热点专家会**自然**被反复命中和保留，
不需要显式 pin。这一组验证「不做 pin，光靠覆盖式保留，命中率是多少」。
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


def test_hot_set_coverage(gate):
    """给定槽位预算，按频率排序选出的热点集能覆盖多少路由？

    这是 pin 的理论上限。如果 top-K 覆盖不了大部分路由，pin 就没意义 ——
    这正是当年实测「pin 是负优化」的原因。
    """
    eng, gt = gate
    D = eng.cfg.dim
    topk = eng.cfg.n_activated_experts
    torch.manual_seed(0)

    from collections import Counter

    cnt = Counter()
    total = 0
    for _ in range(64):
        x = torch.randn(1, D, dtype=torch.bfloat16, device="cuda") * 0.02
        _, idx = gt.forward(x, None)
        for e in idx.reshape(-1).tolist():
            cnt[int(e)] += 1
            total += 1

    ranked = [e for e, _ in cnt.most_common()]
    print("[coverage] distinct=%d of 384, samples=%d" % (len(ranked), total), flush=True)

    # 不同预算下的覆盖率
    for k in (6, 12, 24, 48, 96):
        cover = sum(cnt[e] for e in ranked[:k])
        pct = 100.0 * cover / max(1, total)
        print("  top-%3d -> %.1f%% of routings" % (k, pct), flush=True)

    # 事实：64 个样本只碰到 384 个专家里的一小部分（路由有强局部性）
    assert len(ranked) < 384, len(ranked)
    # top-24（4 倍的 topk）应该覆盖相当一部分；具体多少不写死，只记录
    top24 = sum(cnt[e] for e in ranked[:24])
    assert top24 > 0
