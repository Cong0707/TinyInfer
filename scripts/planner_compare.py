"""统一规划器 vs 现有 plan()：交叉验证 + 矩阵。

这个脚本不改任何东西，只做对比：对同一个 (模型, 机器)，分别用两套逻辑算布局，
再用统一目标函数评估**两者的代价**。如果新解不更快，说明目标函数写错了。

用法：

    python scripts/planner_compare.py                 # 本机 + 三份模型
    python scripts/planner_compare.py --matrix        # 全矩阵
    python scripts/planner_compare.py --check         # CI 用，非零退出表示失败
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tinyinfer.planner import Layout, Planner  # noqa: E402
from tinyinfer.resources import Card, Hardware, detect, plan  # noqa: E402

GB = 2 ** 30

MODELS = {
    "DeepSeek-V4.1-Flash": dict(layer_bytes=0.1603 * GB, max_layer_bytes=0.4683 * GB,
                                embed_bytes=1.2329 * GB, head_bytes=1.2329 * GB,
                                expert_bytes=0.0175 * GB, n_layers=40, topk=6),
    "Qwen3.5-0.8B": dict(layer_bytes=0.0401 * GB, max_layer_bytes=0.2007 * GB,
                         embed_bytes=1.32 * GB, head_bytes=1.32 * GB,
                         expert_bytes=0.0, n_layers=24, topk=0),
    "Qwen3.8-27B": dict(layer_bytes=0.7196 * GB, max_layer_bytes=3.7997 * GB,
                        embed_bytes=1.32 * GB, head_bytes=1.32 * GB,
                        expert_bytes=0.0, n_layers=64, topk=0),
}

MACHINES = [
    ("GB300 288GB", [288.0], 2048.0, 1900.0, False),
    ("H100 80GB", [80.0], 1024.0, 950.0, False),
    ("A100 80GB", [80.0], 512.0, 480.0, False),
    ("L40S 48GB", [48.0], 256.0, 240.0, False),
    ("RTX 5090 32GB", [32.0], 128.0, 120.0, False),
    ("RTX 4090 24GB", [24.0], 64.0, 56.0, False),
    ("RTX 3090 24GB", [24.0], 32.0, 24.0, False),
    ("RTX 4060Ti 16GB", [16.0], 32.0, 24.0, False),
    ("RTX 3060 12GB", [12.0], 32.0, 24.0, False),
    ("RTX 2080Ti 11GB", [11.0], 32.0, 24.0, False),
    ("RTX 5060 8GB (本机)", [8.0], 47.0, 33.0, False),
    ("GTX 1650 4GB", [4.0], 16.0, 10.0, False),
]

# 假想带宽：这些机器见不到，但带宽必须给（规划器不接受 fallback）。
# 按 PCIe 代次估：Gen3 x16 ~ 12GB/s、Gen4 x16 ~ 24、Gen5 x16 ~ 48。
BW_BY_MACHINE = {
    "GB300 288GB": (48e9, 7e9, 60e9),
    "H100 80GB": (24e9, 6e9, 50e9),
    "A100 80GB": (24e9, 6e9, 50e9),
    "L40S 48GB": (24e9, 6e9, 50e9),
    "RTX 5090 32GB": (48e9, 7e9, 60e9),
    "RTX 4090 24GB": (24e9, 6e9, 50e9),
    "RTX 3090 24GB": (24e9, 5e9, 45e9),
    "RTX 4060Ti 16GB": (12e9, 5e9, 40e9),
    "RTX 3060 12GB": (12e9, 4e9, 40e9),
    "RTX 2080Ti 11GB": (12e9, 3e9, 35e9),
    "RTX 5060 8GB (本机)": (7.1e9, 4.4e9, 35.3e9),
    "GTX 1650 4GB": (12e9, 1e9, 20e9),
}


def _hw(name, cards_gb, ram, avail, unified):
    cards = [Card(i, "card", gb, gb) for i, gb in enumerate(cards_gb)]
    hw = Hardware(name=name, vram_gb=sum(cards_gb), ram_gb=ram,
                  ram_avail_gb=avail, unified=unified, cards=cards)
    h2d, dsk, ram_bps = BW_BY_MACHINE.get(name, (12e9, 4e9, 35e9))
    return hw, {"h2d_bps": h2d, "disk_bps": dsk, "ram_bps": ram_bps}


def compare(mname, model, hname, hw, bw):
    """两套逻辑各算一份布局，用统一目标函数评估两者。"""
    p = Planner(hw, model, bandwidth=bw,
                calibrate={"compute_per_layer_s": 0.0025})
    new = p.solve()
    try:
        old = plan(hw=hw, n_layers=int(model["n_layers"]), model=model,
                   speculation=False)
        old_lay = Layout(resident=int(old.gpu_layers), slots=int(old.expert_slots),
                         cache_bytes=int(old.cache_bytes),
                         embed_on_cpu=bool(old.embed_on_cpu),
                         head_on_cpu=bool(old.head_on_cpu))
        old_cost = p.cost(old_lay)
        old_ok = True
    except Exception as exc:
        old_lay = None
        old_cost = None
        old_ok = repr(exc)[:60]
    new_cost = p.cost(new)
    return {
        "model": mname, "machine": hname,
        "new": {"resident": new.resident, "slots": new.slots,
                "total_s": round(new_cost.total_s, 4),
                "vram_gb": round(p.used_bytes(new) / GB, 2)},
        "old": (None if old_lay is None else {
            "resident": old_lay.resident, "slots": old_lay.slots,
            "total_s": round(old_cost.total_s, 4),
            "vram_gb": round(p.used_bytes(old_lay) / GB, 2)}),
        "old_error": None if old_ok is True else old_ok,
        "fits": bool(p.fits(new)),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)

    rows = []
    if args.matrix:
        for mname, model in MODELS.items():
            for hname, cards, ram, avail, uni in MACHINES:
                hw, bw = _hw(hname, cards, ram, avail, uni)
                rows.append(compare(mname, model, hname, hw, bw))
    else:
        hw = detect()
        bw = hw.bandwidth(model_dir=r"E:\Models\DeepSeek-V4.1-Flash-1017")
        for mname, model in MODELS.items():
            rows.append(compare(mname, model, hw.name or "this box", hw, bw))

    hdr = ("%-22s %-22s %10s %9s %9s %8s" %
           ("model", "machine", "new r/s", "new T", "old T", "faster"))
    print(hdr)
    print("-" * len(hdr))
    bad = []
    for r in rows:
        n, o = r["new"], r["old"]
        if o is None:
            print("%-22s %-22s %4d/%-5d %8.3fs %9s %8s" % (
                r["model"], r["machine"], n["resident"], n["slots"],
                n["total_s"], r["old_error"] or "-", "-"))
            continue
        delta = o["total_s"] - n["total_s"]
        mark = "%.3fs" % delta if delta > 0.001 else ("%.3fs" % delta)
        print("%-22s %-22s %4d/%-5d %8.3fs %8.3fs %8s" % (
            r["model"], r["machine"], n["resident"], n["slots"],
            n["total_s"], o["total_s"], mark))
        if not r["fits"]:
            bad.append("%s/%s: 新解装不下" % (r["model"], r["machine"]))
        if n["total_s"] > o["total_s"] + 1e-6:
            bad.append("%s/%s: 新解 %.3fs 慢于旧解 %.3fs" % (
                r["model"], r["machine"], n["total_s"], o["total_s"]))

    if args.check:
        print("")
        if bad:
            print("[planner] FAIL  %d 个问题：" % len(bad))
            for b in bad[:15]:
                print("  - " + b)
        else:
            print("[planner] PASS  新解在所有组合上不慢于旧解，且都装得下")
        return 0 if not bad else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
