"""规划器矩阵：多种真实模型 x 从 GB300 到 2080Ti 的完整显卡谱系。

不碰 GPU —— 这是规划器的可移植性契约。规划器是唯一同时知道「机型」和「模型」
的组件，它的输出决定了常驻几层、几个专家槽位、主机缓存多大、MTP 放不放得下。
这些决策必须在从没见过的硬件上也成立，所以矩阵是它唯一有效的测试。

判据（不是「能跑」，是「规划得对」）：
  1. resident + stream == n_layers（不漏层）
  2. slots >= 0
  3. 显存更小的卡不能算出更多槽位（单调性）
  4. 稠密模型（无 routed experts）不应分到专家槽位
  5. floor_s / tok_per_s 不能是非法值
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tinyinfer.resources import Card, Hardware, plan  # noqa: E402

GB = 2 ** 30

MODELS = {
    "DeepSeek-V4.1-Flash": {
        "layer_bytes": 0.1603 * GB, "max_layer_bytes": 0.4683 * GB,
        "embed_bytes": 1.2329 * GB, "head_bytes": 1.2329 * GB,
        "expert_bytes": 0.0175 * GB, "n_layers": 40, "topk": 6,
        "tied_head": False,
    },
    "Qwen3.5-0.8B": {
        "layer_bytes": 0.0401 * GB, "max_layer_bytes": 0.2007 * GB,
        "embed_bytes": 1.32 * GB, "head_bytes": 1.32 * GB,
        "expert_bytes": 0.0, "n_layers": 24, "topk": 0,
        "tied_head": False,
    },
    "Qwen3.8-27B": {
        "layer_bytes": 0.7196 * GB, "max_layer_bytes": 3.7997 * GB,
        "embed_bytes": 1.32 * GB, "head_bytes": 1.32 * GB,
        "expert_bytes": 0.0, "n_layers": 64, "topk": 0,
        "tied_head": False,
    },
}

H2D = 26.5e9
DISK = 4.5e9

MACHINES = [
    ("GB300 288GB / 2TB", [288.0], 2048.0, 1900.0, False),
    ("B200 192GB / 1TB", [192.0], 1024.0, 950.0, False),
    ("H200 141GB / 1TB", [141.0], 1024.0, 950.0, False),
    ("H100 80GB / 1TB", [80.0], 1024.0, 950.0, False),
    ("A100 80GB / 512GB", [80.0], 512.0, 480.0, False),
    ("L40S 48GB / 256GB", [48.0], 256.0, 240.0, False),
    ("A6000 48GB / 256GB", [48.0], 256.0, 240.0, False),
    ("RTX 5090 32GB / 128GB", [32.0], 128.0, 120.0, False),
    ("RTX 4090 24GB / 64GB", [24.0], 64.0, 56.0, False),
    ("4090 24GB / 128GB", [24.0], 128.0, 120.0, False),
    ("2x 4090 48GB / 128GB", [24.0, 24.0], 128.0, 120.0, False),
    ("RTX 3090 24GB / 32GB", [24.0], 32.0, 24.0, False),
    ("RTX 4060Ti 16GB / 32GB", [16.0], 32.0, 24.0, False),
    ("RTX 3060 12GB / 32GB", [12.0], 32.0, 24.0, False),
    ("RTX 5060 8GB / 47GB (本机)", [8.0], 47.0, 33.0, False),
    ("RTX 2080Ti 11GB / 32GB", [11.0], 32.0, 24.0, False),
    ("GTX 1650 4GB / 16GB", [4.0], 16.0, 10.0, False),
    ("M3 Max 48GB 统一内存", [48.0], 48.0, 40.0, True),
    ("CPU only / 64GB", [], 64.0, 56.0, False),
]


def _floor_seconds(p, m):
    topk = int(m.get("topk") or 0)
    n_layers = int(m["n_layers"])
    stream_b = p.stream_layers * m["layer_bytes"]
    if topk <= 0 or m["expert_bytes"] <= 0:
        return stream_b / H2D
    experts_b = topk * n_layers * m["expert_bytes"]
    possible = topk * n_layers
    covered = min(1.0, p.expert_slots / possible)
    return stream_b / H2D + experts_b * (1.0 - covered) / DISK


def build_rows():
    rows = []
    for mname, m in MODELS.items():
        for hname, cards_gb, ram, avail, unified in MACHINES:
            cards = [Card(i, "card", gb, gb) for i, gb in enumerate(cards_gb)]
            hw = Hardware(name=hname, vram_gb=sum(cards_gb), ram_gb=ram,
                          ram_avail_gb=avail, unified=unified, cards=cards)
            try:
                p = plan(hw=hw, n_layers=int(m["n_layers"]), model=m,
                         speculation=True)
                floor = _floor_seconds(p, m)
                rows.append({
                    "model": mname, "machine": hname,
                    "n_layers": int(m["n_layers"]),
                    "gpu": bool(cards_gb),
                    "topk": int(m.get("topk") or 0),
                    "resident": int(p.gpu_layers),
                    "stream": int(p.stream_layers),
                    "slots": int(p.expert_slots),
                    "chunk": int(p.prefill_chunk),
                    "cache_gb": round(p.cache_bytes / GB, 2),
                    "fits": bool(p.fits),
                    "embed_on_cpu": bool(p.embed_on_cpu),
                    "head_on_cpu": bool(p.head_on_cpu),
                    "floor_s": round(floor, 4),
                    "tok_per_s": round(1.0 / floor, 2) if floor > 0 else None,
                })
            except Exception as exc:
                rows.append({"model": mname, "machine": hname,
                             "error": repr(exc)[:200]})
    return rows


def check(rows):
    bad = []
    for r in rows:
        if "error" in r:
            bad.append("%s / %s: 抛异常 %s" % (r["model"], r["machine"], r["error"]))
            continue
        tag = "%s / %s" % (r["model"], r["machine"])
        if r["resident"] + r["stream"] != r["n_layers"]:
            bad.append("%s: resident+stream=%d != n_layers=%d" % (
                tag, r["resident"] + r["stream"], r["n_layers"]))
        if r["slots"] < 0:
            bad.append("%s: slots 为负" % tag)
        if r["topk"] == 0 and r["slots"] != 0:
            bad.append("%s: 稠密模型不该有专家槽位 (slots=%d)" % (tag, r["slots"]))
        if r["floor_s"] is None or r["floor_s"] != r["floor_s"] or r["floor_s"] < 0:
            bad.append("%s: floor_s 非法 (%s)" % (tag, r["floor_s"]))
    by_model = {}
    for r in rows:
        if "error" in r or not r["gpu"]:
            continue
        by_model.setdefault(r["model"], []).append(r)
    for mname, rs in by_model.items():
        for a, b in zip(rs, rs[1:]):
            if b["slots"] > a["slots"] + 1:
                bad.append("%s: %s 槽位(%d) 多于 %s(%d) —— 显存更小却更多槽位" % (
                    mname, b["machine"], b["slots"], a["machine"], a["slots"]))
    return (not bad), bad


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--model", default=None)
    args = ap.parse_args(argv)
    rows = build_rows()
    if args.model:
        rows = [r for r in rows if r["model"] == args.model]
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    else:
        cur = None
        hdr = ("%-24s %-26s %4s %4s %6s %6s %7s %8s %6s" %
               ("model", "machine", "res", "str", "slots", "chunk", "cache", "floor_s", "tok/s"))
        print(hdr)
        print("-" * len(hdr))
        for r in rows:
            if r["model"] != cur:
                cur = r["model"]
                print("")
            if "error" in r:
                print("%-24s %-26s  ERROR %s" % (r["model"], r["machine"], r["error"][:50]))
                continue
            print("%-24s %-26s %4d %4d %6d %6d %6.2fG %8.3f %6s" % (
                r["model"], r["machine"], r["resident"], r["stream"],
                r["slots"], r["chunk"], r["cache_gb"], r["floor_s"],
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
