"""规划器矩阵：多份真实模型 x 从 GB300 到 GTX 1650 的机器谱系。

不碰 GPU —— 这是规划器的**可移植性契约**。规划器是唯一同时知道「机型」和
「模型」的组件，它的输出决定了常驻几层、几个专家槽位、主机缓存多大。这些决策
必须在从没见过的硬件上也成立，所以矩阵是它唯一有效的测试。

规划只有一个实现：planner.Planner（旧的 resources.plan() 已删除 —— 那是 563 行
启发式 + 12 个实测常数的堆叠，换台机器就不对）。

用法：
    python scripts/plan_matrix.py                # 人读的表格
    python scripts/plan_matrix.py --json         # 机器可读
    python scripts/plan_matrix.py --check        # CI 用，非零退出表示失败
    python scripts/plan_matrix.py --model X      # 只看一份模型
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tinyinfer.planner import Planner  # noqa: E402
from tinyinfer.resources import Card, Hardware  # noqa: E402

GB = 2 ** 30

# ---- 三份真实模型的几何（model_profile 实测）---------------------------------
MODELS = {
    "DeepSeek-V4.1-Flash": dict(layer_bytes=0.1603 * GB, max_layer_bytes=0.4683 * GB,
                                embed_bytes=1.2329 * GB, head_bytes=1.2329 * GB,
                                expert_bytes=0.0175 * GB, n_layers=40, topk=6,
                                n_experts=384),
    "Qwen3.5-0.8B": dict(layer_bytes=0.0401 * GB, max_layer_bytes=0.2007 * GB,
                         embed_bytes=1.32 * GB, head_bytes=1.32 * GB,
                         expert_bytes=0.0, n_layers=24, topk=0),
    "Qwen3.8-27B": dict(layer_bytes=0.7196 * GB, max_layer_bytes=3.7997 * GB,
                        embed_bytes=1.32 * GB, head_bytes=1.32 * GB,
                        expert_bytes=0.0, n_layers=64, topk=0),
}

# ---- 机器谱系：数据中心卡到 2018 年的消费卡 -----------------------------------
# 带宽按 PCIe 代次估（h2d, disk, ram），单位 byte/s。
# 规划器不接受带宽 fallback —— 假想机器也得给它一个数。
MACHINES = [
    ("GB300 288GB", [288.0], 2048.0, 1900.0, False, (48e9, 7e9, 60e9)),
    ("B200 192GB", [192.0], 1024.0, 950.0, False, (48e9, 7e9, 60e9)),
    ("H200 141GB", [141.0], 1024.0, 950.0, False, (24e9, 6e9, 50e9)),
    ("H100 80GB", [80.0], 1024.0, 950.0, False, (24e9, 6e9, 50e9)),
    ("A100 80GB", [80.0], 512.0, 480.0, False, (24e9, 6e9, 50e9)),
    ("L40S 48GB", [48.0], 256.0, 240.0, False, (24e9, 6e9, 50e9)),
    ("A6000 48GB", [48.0], 256.0, 240.0, False, (24e9, 6e9, 50e9)),
    ("RTX 5090 32GB", [32.0], 128.0, 120.0, False, (48e9, 7e9, 60e9)),
    ("RTX 4090 24GB", [24.0], 64.0, 56.0, False, (24e9, 6e9, 50e9)),
    ("2x 4090 48GB", [24.0, 24.0], 128.0, 120.0, False, (24e9, 6e9, 50e9)),
    ("RTX 3090 24GB", [24.0], 32.0, 24.0, False, (24e9, 5e9, 45e9)),
    ("RTX 4060Ti 16GB", [16.0], 32.0, 24.0, False, (12e9, 5e9, 40e9)),
    ("RTX 3060 12GB", [12.0], 32.0, 24.0, False, (12e9, 4e9, 40e9)),
    ("RTX 5060 8GB (本机)", [8.0], 47.0, 33.0, False, (7.1e9, 4.4e9, 35.3e9)),
    ("RTX 2080Ti 11GB", [11.0], 32.0, 24.0, False, (12e9, 3e9, 35e9)),
    ("GTX 1650 4GB", [4.0], 16.0, 10.0, False, (12e9, 1e9, 20e9)),
    ("M3 Max 48GB 统一内存", [48.0], 48.0, 40.0, True, (30e9, 6e9, 100e9)),
]


def build_rows(only_model=None):
    rows = []
    for mname, m in MODELS.items():
        if only_model and mname != only_model:
            continue
        for hname, cards_gb, ram, avail, unified, bw in MACHINES:
            cards = [Card(i, "card", gb, gb) for i, gb in enumerate(cards_gb)]
            hw = Hardware(name=hname, vram_gb=sum(cards_gb), ram_gb=ram,
                          ram_avail_gb=avail, unified=unified, cards=cards)
            bandwidth = {"h2d_bps": bw[0], "disk_bps": bw[1], "ram_bps": bw[2]}
            try:
                p = Planner(hw, m, bandwidth=bandwidth)
                lay = p.solve()
                c = p.cost(lay)
                rows.append({
                    "model": mname, "machine": hname,
                    "n_layers": int(m["n_layers"]),
                    "gpu": bool(cards_gb),
                    "resident": int(lay.resident),
                    "stream": int(lay.stream_layers),
                    "slots": int(lay.slots),
                    "cache_gb": round(lay.cache_bytes / GB, 2),
                    "fits": bool(p.used_bytes(lay) <= p.vram),
                    "vram_used_gb": round(p.used_bytes(lay) / GB, 2),
                    "floor_s": round(c.total_s, 4),
                    "tok_per_s": round(1.0 / c.total_s, 2) if c.total_s > 0 else None,
                    "reason": lay.reason,
                })
            except Exception as exc:
                rows.append({"model": mname, "machine": hname,
                             "n_layers": int(m["n_layers"]),
                             "error": repr(exc)[:200]})
    return rows


def check(rows):
    """可移植性判据。返回 (ok, 问题列表)。"""
    bad = []
    for r in rows:
        if "error" in r:
            bad.append("%s / %s: 抛异常 %s" % (r["model"], r["machine"], r["error"]))
            continue
        tag = "%s / %s" % (r["model"], r["machine"])
        if r["resident"] + r["stream"] != r["n_layers"]:
            bad.append("%s: resident+stream=%d != n_layers=%d" % (
                tag, r["resident"] + r["stream"], r["n_layers"]))
        if r["slots"] < 0 or r["resident"] < 0:
            bad.append("%s: 负数 (resident=%d slots=%d)" % (
                tag, r["resident"], r["slots"]))
        # 「装不下」本身不是失败 —— 只要求解器如实报告（resident=0 + fits=False）
        # 而不是给出一个自相矛盾的布局。真正的失败是：说能装下但算出来超了。
        if not r["fits"] and (r["resident"] > 0 or r["slots"] > 0):
            bad.append("%s: 报告 fits=True 但布局超了 (resident=%d slots=%d 用 %.2fGB)"
                       % (tag, r["resident"], r["slots"], r["vram_used_gb"]))
        if r["floor_s"] is None or r["floor_s"] < 0 or r["floor_s"] != r["floor_s"]:
            bad.append("%s: floor_s 非法 (%s)" % (tag, r["floor_s"]))
    # 同一模型下，显存越大常驻层不该更少
    by_model = {}
    for r in rows:
        if "error" in r or not r["gpu"]:
            continue
        by_model.setdefault(r["model"], []).append(r)

    def _vram(name):
        for hname, cards_gb, *_ in MACHINES:
            if hname == name:
                return sum(cards_gb)
        return 0.0

    for mname, rs in by_model.items():
        rs.sort(key=lambda x: -_vram(x["machine"]))
        for a, b in zip(rs, rs[1:]):
            if _vram(a["machine"]) == _vram(b["machine"]):
                continue
            if a["resident"] + 1 < b["resident"]:
                bad.append("%s: %s(%.0fGB) 常驻 %d 少于 %s(%.0fGB) 的 %d" % (
                    mname, a["machine"], _vram(a["machine"]), a["resident"],
                    b["machine"], _vram(b["machine"]), b["resident"]))
    return (not bad), bad


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--model", default=None)
    args = ap.parse_args(argv)

    rows = build_rows(args.model)

    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    else:
        cur = None
        hdr = "%-24s %-24s %5s %5s %7s %8s %9s %8s" % (
            "model", "machine", "res", "str", "slots", "vram", "floor_s", "tok/s")
        print(hdr)
        print("-" * len(hdr))
        for r in rows:
            if r["model"] != cur:
                cur = r["model"]
                print("")
            if "error" in r:
                print("%-24s %-24s  ERROR %s" % (r["model"], r["machine"], r["error"][:40]))
                continue
            print("%-24s %-24s %5d %5d %7d %7.2fG %9.3f %8s" % (
                r["model"], r["machine"], r["resident"], r["stream"],
                r["slots"], r["vram_used_gb"], r["floor_s"],
                r["tok_per_s"] if r["tok_per_s"] is not None else "-"))

    if args.check:
        ok, bad = check(rows)
        print("")
        if ok:
            print("[planner] PASS  %d 个组合全部满足可移植性判据" % len(rows))
        else:
            print("[planner] FAIL  %d 个问题：" % len(bad))
            for b in bad[:20]:
                print("  - " + b)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
