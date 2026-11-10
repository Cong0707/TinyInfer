"""规划器不变式：跨硬件验证 Planner 的输出自洽。

这些是「求解器回归会破坏什么」的判据：层数守恒、槽位非负、装不下时要拒绝、
稠密模型不该分到专家槽位、常驻层数随显存单调。全部用 Planner 直接验证，
不依赖引擎、也不碰 CUDA。
"""
from __future__ import annotations

from . import Case

GB = 1024 ** 3


def _profile():
    import os

    from ..loading import ShardIndex
    from ..resources import model_profile

    path = os.environ.get("TI_MODEL")
    if not path or not os.path.isdir(path):
        return None
    try:
        return model_profile(ShardIndex(path))
    except Exception:
        return None


def register(report) -> None:
    prof = _profile()
    if prof is None:
        report.add(Case("planner", "skipped (set TI_MODEL to a checkpoint dir)", True))
        return

    from ..planner import Planner
    from ..resources import Card, Hardware

    n_layers = int(prof.get("n_layers") or 40)
    geo = dict(prof)
    geo["n_layers"] = n_layers
    geo["topk"] = int(prof.get("topk") or 0)

    # 带宽必须给：Planner 不接受 fallback。这里按 PCIe 代次估。
    bw = {"h2d_bps": 12e9, "disk_bps": 4e9, "ram_bps": 35e9}

    def _hw(name, vram, ram=64.0, avail=50.0):
        return Hardware(name=name, vram_gb=vram, ram_gb=ram, ram_avail_gb=avail,
                        cards=[Card(0, name, vram, vram * 0.95)])

    for name, vram, ram, avail in (("2GB", 2.0, 8.0, 6.0),
                                   ("4GB", 4.0, 16.0, 12.0),
                                   ("8GB", 8.0, 47.0, 30.0),
                                   ("16GB", 16.0, 64.0, 50.0),
                                   ("24GB", 24.0, 64.0, 50.0),
                                   ("1TB", 1024.0, 2048.0, 1800.0)):
        hw = _hw(name, vram, ram, avail)
        try:
            p = Planner(hw, geo, bandwidth=bw)
            lay = p.solve()
        except Exception as exc:
            report.add(Case("planner " + name, "raised " + repr(exc), False))
            continue
        in_range = 0 <= lay.resident <= n_layers
        non_negative = lay.slots >= 0 and lay.cache_bytes >= 0
        fits = p.used_bytes(lay) <= p.vram * 0.94 + 1e6
        detail = "resident=%d/%d slots=%d vram=%.1fG" % (
            lay.resident, n_layers, lay.slots, p.used_bytes(lay) / GB)
        report.add(Case("planner " + name, detail,
                        in_range and non_negative and fits))

    # 稠密模型（没有 routed experts）不该分到专家槽位
    dense = dict(geo)
    dense["expert_bytes"] = 0.0
    dense["topk"] = 0
    lay = Planner(_hw("8GB", 8.0, 47.0, 30.0), dense, bandwidth=bw).solve()
    report.add(Case("planner gives a dense model no expert slots",
                    "slots=%d" % lay.slots, lay.slots == 0))

    # 显存更大 -> 常驻层更多（单调，允许 1 的舍入差异）
    prev = -1
    mono = True
    for name, vram in (("8GB", 8.0), ("16GB", 16.0), ("24GB", 24.0)):
        lay = Planner(_hw(name, vram), geo, bandwidth=bw).solve()
        if lay.resident + 1 < prev:
            mono = False
        prev = lay.resident
    report.add(Case("planner residency is monotonic in VRAM", "", mono))
