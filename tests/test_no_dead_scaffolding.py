"""扫一遍「算出来没人用」的字段：这是本工程反复出现的陷阱。

已经发现的实例：

  - ServingEngine 的 KVPool（文档说有，类里从来没有）
  - engine._expert_dev（静态专家分配，decode 路径从不读）
  - resources.Plan.card_slots（算出来没有消费者）
  - moe.py 不读 ep_world / ep_rank（TI_EP=1 是空操作）

这些的共同点：读代码时看起来设施齐备，跑起来什么也没发生。这一组测试不是
测功能，是测「声称存在的东西真的存在」，防止文档和实现对不上。
"""

import re

import pytest

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "src" / "tinyinfer"


def _read(rel):
    p = ROOT / rel
    return p.read_text(encoding="utf-8") if p.exists() else ""


def test_serving_engine_really_has_the_pool_its_docstring_promises():
    """文档说 KVPool 决定一步能装几条，就得真的有。"""
    src = _read("sched.py")
    assert "KVPool" in src
    assert "self.pool = KVPool(" in src, "ServingEngine lost its KVPool"


def test_ep_is_flagged_when_unwired():
    """moe.py 不读 ep_world 时，engine 设置它就等于空操作，必须说明。"""
    moe = _read("deepseek/moe.py")
    engine = _read("engine.py")
    moe_uses_ep = ("ep_world" in moe) or ("ep_rank" in moe)
    engine_sets_ep = "ffn.ep_world" in engine
    if engine_sets_ep and not moe_uses_ep:
        markers = ("no-op", "not implemented", "空操作", "WARNING")
        assert any(m in engine for m in markers), (
            "engine sets ep_world but moe.py never reads it, and nothing says so")
    else:
        pytest.skip("EP is wired up now -- flip this test to a positive assertion")


def test_loader_calls_only_defined_index_methods():
    """loader 用到的每个 self.index.X() 都必须真的定义在 ShardIndex 上。"""
    loader = _read("loading.py")
    used = set(re.findall(r"self\.index\.([a-zA-Z_][a-zA-Z0-9_]*)\(", loader))
    if not used:
        pytest.skip("no index calls found")
    # ShardIndex 定义在 loading.py 自己里（不是 weights.py）
    defined = set(re.findall(r"def ([a-zA-Z_][a-zA-Z0-9_]*)\(", loader))
    missing = {m for m in used if m not in defined}
    assert not missing, "loader calls undefined ShardIndex methods: %s" % sorted(missing)
