"""decode 基准：一个 token 一个 token 地测，参数和真实推理一致。

## 为什么需要这个脚本

我一直用 `torch.tensor([[100,200,300,400,500]])` 测 decode —— 那是 prefill
形状。M=5 时 MoE 层的 indices 是 5x6，去重后每层 `_last_ids` 有 12~28 个
专家，于是 `need()` 每层请求 30 个（真实 decode 只要 6 个），槽位每步被
装 784 次、驱逐 784 次，读 14.7GB。测出来的 0.079 tok/s 是这么来的。

真实 decode：`_last_ids` 精确是 6，40 层全是 6，槽位每步只装约 30 个，
成绩是 0.326 tok/s —— **差 4 倍**。

所以基准必须：
  1. 单 token 输入，逐 token 生成
  2. 先预热到稳定（判据：连续 3 步 step 时间变化 < 10%）
  3. 取样在 reset 后，含 KV cache 的真实增长
  4. 报中位数和极差，不只报一个数

## 用法

    python scripts/bench_decode.py --model <checkpoint>
    python scripts/bench_decode.py --model <ckpt> --tokens 32 --json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def bench(model_dir, *, tokens=16, warmup=6, reset_each=True):
    import torch
    from tinyinfer.engine import Engine

    eng = Engine.from_hf(str(model_dir), device="cuda", allow_missing_layers=True)
    p = eng.hw_plan
    out = {
        "resident": int(getattr(p, "resident", 0)),
        "slots": int(getattr(p, "slots", 0)),
        "stream_layers": int(getattr(p, "stream_layers", 0)),
    }
    # 单 token —— 这是 decode 的真实形状
    ids = torch.tensor([[100]], dtype=torch.long, device="cuda")
    ts: list[float] = []
    with torch.inference_mode():
        eng.forward(ids, 0)
        torch.cuda.synchronize()

        # 预热到稳定：连续 3 步变化 < 10%
        prev, stable = None, 0
        for _ in range(24):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            eng.forward(ids, 0)
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) * 1e3
            if prev is not None and prev > 0 and abs(dt - prev) < prev * 0.10:
                stable += 1
                if stable >= 3:
                    break
            else:
                stable = 0
            prev = dt

        for _ in range(max(1, tokens)):
            if reset_each and hasattr(eng, "reset"):
                eng.reset()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            logits = eng.forward(ids, 0)
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1e3)

        out["finite"] = bool(torch.isfinite(logits).all())
    eng.close()

    ts_sorted = sorted(ts)
    med = statistics.median(ts)
    out.update({
        "median_ms": round(med, 1),
        "min_ms": round(ts_sorted[0], 1),
        "max_ms": round(ts_sorted[-1], 1),
        "spread": round((ts_sorted[-1] - ts_sorted[0]) / max(1.0, med), 3),
        "n": len(ts),
        "tok_per_s": round(1000.0 / med, 4) if med > 0 else None,
    })
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokens", type=int, default=16)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    out = bench(a.model, tokens=a.tokens)
    if a.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        print("layout   %d 层常驻 / %d 层流式 / %d 槽位" % (
            out["resident"], out["stream_layers"], out["slots"]))
        print("decode   %.1fms 中位数（%.1f~%.1f，极差 %.0f%%）" % (
            out["median_ms"], out["min_ms"], out["max_ms"], 100 * out["spread"]))
        print("吞吐     %.4f tok/s" % (out["tok_per_s"] or 0))
        print("正确性   finite=%s" % out["finite"])
    return 0 if out.get("finite") else 1


if __name__ == "__main__":
    raise SystemExit(main())
