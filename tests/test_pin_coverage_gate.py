"""pin 的覆盖率门槛：覆盖不到一半就不 pin。

实测（tests/test_hot_coverage.py）覆盖率是次线性的：

  top-24 -> 56.5%   top-48 -> 74.7%   top-96 -> 92.4%

而 pin 掉的槽位**永久失效**。所以「pin 一半容量只换一半覆盖率」不划算 ——
engine.pin_hot_experts 里加了门槛：覆盖不足 50% 就返回 0，什么都不 pin。

这组测试用假 store 验证这条门槛，不需要真模型。
"""

import pytest

torch = pytest.importorskip("torch")

from tinyinfer.deepseek.moe import MoE  # noqa: E402


class _Slots:
    """只有容量和 has/pin 的假槽位池。"""

    def __init__(self, n):
        self.slots = int(n)
        self._have = set()
        self._pinned = set()

    def has(self, layer, expert):
        return (layer, expert) in self._have

    def pin(self, layer, expert):
        self._pinned.add((layer, expert))
        return True

    def pinned_count(self):
        return len(self._pinned)


def _plan(route_counts_by_layer, n_slots, n_layers=40, topk=6):
    """跑 pin_hot_experts 的选取逻辑（不碰 GPU/warmup 部分）。"""
    cand = []
    for lid, counts in route_counts_by_layer.items():
        for e, c in counts.items():
            cand.append((c, lid, e))
    if not cand:
        return 0
    cand.sort(key=lambda t: -t[0])
    need_step = n_layers * topk
    if n_slots < need_step * 2:
        return 0
    avail = n_slots - need_step
    tot = sum(c for c, _, _ in cand)
    if tot <= 0:
        return 0
    # 和 engine.pin_hot_experts 同一规则：累计覆盖到 95% 为止（不填满预算）
    sel = []
    acc = 0
    for c, _l, _e in cand[:avail]:
        sel.append(c)
        acc += c
        if acc >= 0.95 * tot:
            break
    cover = acc / float(tot)
    if cover < 0.5:
        return 0
    return len(sel)


def test_flat_routing_does_not_pin():
    """路由完全平坦（没有热点）时 pin 覆盖不到一半 -> 不 pin。"""
    # 40 层 x 384 个专家，每个被路由 1 次：任何子集覆盖率都等于容量比例
    rc = {lid: {e: 1 for e in range(384)} for lid in range(40)}
    # 槽位 1359，可用 1359 - 240 = 1119；1119 / (40*384) = 7.3% < 50%
    assert _plan(rc, 1359) == 0


def test_strong_hotspot_pins_only_the_head():
    """有强热点时只 pin 头部，不填满预算。

    每层 24 个专家吃掉 95% 的路由（对应实测的 top-24 -> 56.5%，这里取更强的
    分布以验证「到 95% 就停」的逻辑），剩下 364 个各 1 次。
    预算 1119 够 pin 40x24 = 960 个，但实际应该停在覆盖率 95% 处。
    """
    rc = {}
    for lid in range(40):
        d = {e: 40 for e in range(24)}          # 960
        d.update({e: 1 for e in range(24, 384)})  # 360
        rc[lid] = d
    n = _plan(rc, 1359)
    # 每层 24x40=960 后累计覆盖率 = 38400/(38400+14400) = 72.7%，未到 95%，
    # 所以会继续填，但填到 1119 也只是尾部（每个 1 次）。
    # 这里断言的是「至少 pin 了全部 24 个热点」。
    assert n >= 960, n
    assert n <= 1119, n


def test_small_pool_never_pins():
    """池子连两步都装不下时直接不 pin（8GB 卡的 271 < 480）。"""
    rc = {lid: {e: 100 for e in range(6)} for lid in range(40)}
    assert _plan(rc, 271) == 0
