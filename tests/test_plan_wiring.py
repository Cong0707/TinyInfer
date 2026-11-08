"""规划器的每个字段都必须真的被消费，或明确标注为「仅供参考」。

这一类问题在本工程里反复出现，每次都付出真金白银的代价：

  - engine 硬编码 `embed_on_cpu = False`，而 plan 按 True 记账
    -> 多占 1.23GB，8GB 卡上白丢 5 层常驻（free 0.33GB -> 1.50GB）
  - ServingEngine 文档说 KVPool 决定批宽，类里从来没有
  - moe.py 不读 ep_world，TI_EP=1 是空操作
  - resources.Plan.card_slots 没有消费者

这里用静态检查锁住 Plan 的每个字段：要么在 engine/server 里被读，
要么在 dataclass 注释里写明「仅供外部/参考」。新加字段忘了接线会失败。
"""

import re

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "src" / "tinyinfer"


def _read(rel):
    p = ROOT / rel
    return p.read_text(encoding="utf-8") if p.exists() else ""


def _plan_fields():
    """Plan dataclass 的字段名（按源码顺序）。"""
    src = _read("resources.py")
    m = re.search(r"class Plan:\n(.*?)\n\n", src, re.S)
    assert m, "Plan dataclass not found"
    body = m.group(1)
    return re.findall(r"^\s{4}([a-z_][a-z0-9_]*):", body, re.M)


def test_every_plan_field_is_consumed_or_documented():
    fields = _plan_fields()
    assert fields, "no Plan fields parsed"
    consumers = (_read("engine.py") + _read("server.py")
                 + _read("loading.py") + _read("deepseek/moe.py"))
    plan_src = _read("resources.py")
    unconsumed = []
    for f in fields:
        # 三种读法都算消费：p.<f>、getattr(plan, "<f>")、以及单独出现的名字
        if (re.search(r"\.%s\b" % re.escape(f), consumers)
                or re.search(r"['\"]%s['\"]" % re.escape(f), consumers)):
            continue
        # 允许：dataclass 里该字段上方 6 行内有「参考/外部/未使用」说明
        idx = plan_src.find(f + ":")
        ctx = plan_src[max(0, idx - 400):idx] if idx > 0 else ""
        if any(k in ctx for k in ("没有消费者", "仅供参考", "reference only",
                                  "unused", "外部调用")):
            continue
        unconsumed.append(f)
    assert not unconsumed, (
        "Plan fields nobody reads: %s -- wire them up or document them" % unconsumed)


def test_embed_and_head_placement_comes_from_the_plan():
    """engine 不能硬编码 embed/head 放哪。"""
    eng = _read("engine.py")
    # embed_on_cpu 必须由 hw_plan 推导，不能直接 = False
    assert "self.embed_on_cpu = False" not in eng or \
        "TI_EMBED_GPU" in eng, "embed placement is hardcoded without an escape hatch"
    assert "hw_plan" in eng
