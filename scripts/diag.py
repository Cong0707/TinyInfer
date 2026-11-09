"""统一调试入口：一次运行拿到排查所需的所有证据。

用法：

    python scripts/diag.py --model <模型目录> [--steps 8] [--json out.json]

## 为什么是「一次」而不是「多个开关分别跑」

旧做法要这样查一个问题：

    TI_MEM_TRACE=1   python ...    # 看显存
    TI_MOE_PROF=1    python ...    # 看 MoE 分段
    TI_CACHE_TRACE=1 python ...    # 看缓存
    TI_TIME=1        python ...    # 看时间
    TI_DEBUG_NAN=all python ...    # 看是否 NaN

跑五次、拼五份输出。问题不只是慢：**每次运行的环境状态不一样**（预热进度、
页缓存温度、显存碎片），拼起来的"完整证据"其实来自几个不可比的状态。

本脚本一次跑完，所有观测走 observability.collect_diag()，采样固定，
所以多次运行之间的差异只来自被测对象本身。

## 输出什么

- 硬件与当前生效的调优参数（tuning.json）
- 规划器给出的配置（槽位、缓存、池子、读到几线程）
- 显存：已分配/保留/峰值/剩余/分配重试次数
- 传输：异步 DMA 走了几次、退化几次、多少字节
- 缓存：显存命中率、内存命中率、磁盘次数
- 计算：MoE 各段累计耗时
- **正确性：输出是否有限，同一输入两次结果是否一致**（这条比速度重要）
- 时间：中位数/最小/最大/极差
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=6)
    ap.add_argument("--json", default=None, help="把完整证据写成 JSON")
    ap.add_argument("--env", action="append", default=[], help="KEY=VALUE，可重复")
    a = ap.parse_args(argv)

    for kv in a.env:
        if "=" in kv:
            k, v = kv.split("=", 1)
            os.environ[k] = v

    src = str(Path(__file__).resolve().parents[1] / "src")
    if src not in sys.path:
        sys.path.insert(0, src)

    import torch

    from tinyinfer.engine import Engine
    from tinyinfer.observability import collect_diag, format_diag

    t_build = time.perf_counter()
    eng = Engine.from_hf(a.model, device="cuda", allow_missing_layers=True)
    build_s = time.perf_counter() - t_build

    ids = torch.tensor([[100, 200, 300, 400, 500]], dtype=torch.long, device="cuda")
    steps = []
    correctness = {"finite": None, "repeatable": None, "max_diff": None}

    with torch.inference_mode():
        eng.forward(ids, 0)
        torch.cuda.synchronize()
        _pos = int(ids.shape[1])
        for _ in range(a.warmup):
            eng.forward(torch.tensor([[7]], device="cuda"), _pos)
            torch.cuda.synchronize()
            _pos += 1
        for _ in range(a.steps):
            t0 = time.perf_counter()
            eng.forward(ids, 0)
            torch.cuda.synchronize()
            steps.append((time.perf_counter() - t0) * 1e3)

        # 正确性：同一输入两次。这是最便宜也最有价值的检查 ——
        # 一个更快的配置如果输出不确定，它的速度数字没有意义。
        x = eng.forward(ids, 0)
        torch.cuda.synchronize()
        y = eng.forward(ids, 0)
        torch.cuda.synchronize()
        fin = bool(torch.isfinite(x).all()) and bool(torch.isfinite(y).all())
        if fin:
            diff = float((x.float() - y.float()).abs().max())
            scale = float(x.float().abs().max()) + 1e-6
            correctness = {"finite": True, "repeatable": diff <= 1e-2 * scale,
                           "max_diff": round(diff, 6), "scale": round(scale, 3)}
        else:
            correctness = {"finite": False, "repeatable": False,
                           "max_diff": None, "scale": None}

    steps.sort()
    timing = {
        "build_s": round(build_s, 2),
        "steps": len(steps),
        "median_ms": round(statistics.median(steps), 1) if steps else None,
        "min_ms": round(steps[0], 1) if steps else None,
        "max_ms": round(steps[-1], 1) if steps else None,
        "spread": round((steps[-1] - steps[0]) / max(1.0, statistics.median(steps)), 3)
        if steps else None,
    }

    diag = collect_diag(eng, samples=correctness, forward_ms=timing["median_ms"])
    diag["timing"].update(timing)
    diag["build_s"] = round(build_s, 2)

    print(format_diag(diag))
    print("  build    : %.1fs" % build_s)
    print("  timing   : median=%.0fms min=%.0f max=%.0f spread=%.2f" % (
        timing["median_ms"] or 0, timing["min_ms"] or 0,
        timing["max_ms"] or 0, timing["spread"] or 0))
    if correctness["finite"]:
        print("  correct  : finite=yes repeatable=%s (diff=%s)" % (
            correctness["repeatable"], correctness["max_diff"]))
    else:
        print("  correct  : finite=NO  <-- 这个配置不可用，速度数字无意义")

    if a.json:
        try:
            Path(a.json).write_text(
                json.dumps(diag, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8")
            print("  json     : " + a.json)
        except Exception as exc:
            print("  json     : failed %r" % (exc,))

    try:
        eng.close()
    except Exception:
        pass
    return 0 if correctness["finite"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
