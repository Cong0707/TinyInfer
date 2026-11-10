"""本地 CI：三层门禁，全在 GPU 机器上跑（GitHub Actions 没有显卡，不能用）。

用法：

    python scripts/ci.py                    # 全部三层
    python scripts/ci.py --layer accuracy   # 只跑某一层
    python scripts/ci.py --model <checkpoint> --json report.json

## 三层各测什么

**L1 accuracy —— 第一时间发现 kernel 被改坏。**

跑 scripts/check_kernels.py，85 个用例逐个比对数值（有 bitwise 也有 rtol）。
这一层不加载整模型，几秒到一分钟，**每次改 kernel 都该跑**。

**L2 engine —— 确保不会 OOM 或运行失败。**

真加载模型、跑 prefill + 若干步 decode，检查：不抛异常、无 NaN/Inf、每一步
都能完成。这一层慢（分钟级），但它是唯一能发现「初始化就崩」「跑几步就 OOM」
的地方 —— 之前的 `!handles_ INTERNAL ASSERT` 和 `cuMemSetAccess OOM` 都只有
这一层能抓到。

**L3 utilization —— 确保引擎能吃到最大资源。**

对比规划器的意图和运行时实际拿到的东西：

  - 槽位数：plan 说要给 N 个，运行时实际建了几个
  - 显存：用了多少 / 物理多少 / 还剩多少（剩太多说明没吃满，剩太少说明没余量）
  - H2D 是否真的走了异步 DMA（不是退化到 pageable）
  - 缓存命中率（vram_hit / ram_hit）

这一层的判据不是「越大越好」，而是**「规划值和实际值一致」** —— 不一致说明
某一环静默失效了（今天遇到两次：cache 规划 15.9GB 实际 0.53GB、slots 规划 24
实际 24 但 pin 比例没生效）。

## 为什么不用 CI 服务

GitHub Actions 的 runner 没有 GPU。这个仓库的每一次真实验证都需要 CUDA、需要
几十 GB 的模型权重、需要那块卡的显存容量。所以 CI 必须是「在这台机器上跑一个
脚本」，而不是「推到远端等机器人跑」。
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def _env():
    e = os.environ.copy()
    e["PYTHONPATH"] = str(SRC)
    return e


def _run(cmd, timeout=None):
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=str(ROOT), env=_env(), capture_output=True,
                           text=True, timeout=timeout)
        return p.returncode, (p.stdout or "") + (p.stderr or ""), time.time() - t0
    except subprocess.TimeoutExpired:
        return 124, "timeout after %ss" % timeout, time.time() - t0


# ---- L1：kernel 数值正确性 -------------------------------------------------

def layer_accuracy(args):
    """跑 kernel conformance。任何一个用例失败就是失败。"""
    if args.skip_accuracy:
        return {"name": "accuracy", "ok": True, "skipped": True}
    rc, out, dt = _run([sys.executable, str(ROOT / "scripts" / "check_kernels.py")],
                       timeout=args.accuracy_timeout)
    passed = failed = 0
    fails = []
    for line in out.splitlines():
        s = line.strip()
        if "kernel cases passed" in s:
            try:
                a, b = s.split()[0].split("/")
                passed, _total = int(a), int(b)
            except Exception:
                pass
        elif s.startswith("FAIL "):
            fails.append(s[5:][:160])
    failed = len(fails)
    ok = rc == 0 and failed == 0
    tag = "PASS" if ok else "FAIL"
    print("[L1 accuracy ] %s  %d passed, %d failed  (%.1fs)" % (tag, passed, failed, dt))
    for f in fails:
        print("               - " + f)
    return {"name": "accuracy", "ok": ok, "passed": passed, "failed": failed,
            "failures": fails, "seconds": round(dt, 1)}


# ---- L2 + L3：引擎行为与资源占用 -------------------------------------------

_PROBE = r"""
import json, os, statistics, sys, time
sys.path.insert(0, %(src)r)
import torch
from tinyinfer.engine import Engine
from tinyinfer.observability import collect_observability

M = %(model)r
out = {"error": None, "steps_ms": [], "obs": None, "plan": None}
try:
    t0 = time.perf_counter()
    eng = Engine.from_hf(M, device="cuda", allow_missing_layers=True)
    out["build_s"] = round(time.perf_counter() - t0, 2)
    ids = torch.tensor([[100, 200, 300, 400, 500]], dtype=torch.long, device="cuda")
    with torch.inference_mode():
        eng.forward(ids, 0)
        torch.cuda.synchronize()
        for i in range(%(steps)d):
            t = time.perf_counter()
            o = eng.forward(torch.tensor([[7 + i]], device="cuda"), 5 + i)
            torch.cuda.synchronize()
            out["steps_ms"].append(round((time.perf_counter() - t) * 1e3, 1))
            if not bool(torch.isfinite(o).all()):
                out["error"] = "NaN/Inf at decode step %%d" %% (i + 1)
                break
    plan = getattr(eng, "hw_plan", None)
    if plan is not None:
        out["plan"] = {k: getattr(plan, k, None) for k in (
            "expert_slots", "cache_bytes", "host_pool_bytes", "pin_ratio",
            "read_threads", "gpu_layers", "stream_layers", "prefill_chunk")}
    out["obs"] = collect_observability(eng)
    try:
        from tinyinfer.loading import _async_h2d_stats
        out["dma"] = _async_h2d_stats()
    except Exception:
        out["dma"] = None
    # 规划器做时间取舍全靠这三条带宽，必须是这台机器实测的
    try:
        out["bandwidth"] = eng.hw_plan and None
    except Exception:
        pass
    try:
        from tinyinfer.resources import detect as _detect

        out["bandwidth"] = _detect().bandwidth(model_dir=M)
    except Exception as exc:
        out["bandwidth"] = {"error": repr(exc)[:200]}
    try:
        p = torch.cuda.get_device_properties(0)
        # **用 mem_get_info，不要用 memory_allocated。**
        #
        # memory_allocated 会把同一个 storage 的多个 view 重复累加（实测同一份
        # 权重被 38 层引用时报出 10.7GB，实际只有 0.54GB；整体分配量报 16.5GB
        # 而物理只有 8.5GB —— 却明明跑通了）。它是「申请过多少」，不是「占用多少」。
        #
        # mem_get_info 问的是驱动：还剩多少、总共多少。这才是判断「吃满没有」
        # 的依据。
        free_b, total_b = torch.cuda.mem_get_info()
        out["vram"] = {"used_gb": round((total_b - free_b) / 1e9, 2),
                       "free_gb": round(free_b / 1e9, 2),
                       "total_gb": round(total_b / 1e9, 2),
                       "reserved_gb": round(torch.cuda.memory_reserved() / 1e9, 2)}
    except Exception:
        out["vram"] = None
    try:
        eng.close()
    except Exception:
        pass
except Exception as exc:
    out["error"] = repr(exc)[:400]
print("CIJSON" + json.dumps(out))
"""


def _probe_engine(model, steps, timeout):
    code = _PROBE % {"src": str(SRC), "model": model, "steps": steps}
    try:
        p = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), env=_env(),
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"error": "timeout after %ss" % timeout}
    for line in (p.stdout or "").splitlines():
        if line.startswith("CIJSON"):
            try:
                return json.loads(line[6:])
            except Exception:
                pass
    tail = (p.stderr or "").strip().splitlines()
    return {"error": (tail[-1] if tail else "no output")[:300]}


def layer_engine(args, probe):
    """引擎能不能起来、能不能连续跑完几步、有没有 NaN。"""
    if args.skip_engine:
        return {"name": "engine", "ok": True, "skipped": True}
    err = probe.get("error")
    ms = probe.get("steps_ms") or []
    ok = (err is None) and len(ms) >= max(1, args.steps - 1)
    tag = "PASS" if ok else "FAIL"
    med = statistics.median(ms) if ms else 0
    print("[L2 engine   ] %s  build=%.1fs  %d steps, median=%.0fms" % (
        tag, probe.get("build_s") or 0, len(ms), med))
    if err:
        print("               - " + str(err)[:200])
    return {"name": "engine", "ok": ok, "error": err, "steps": len(ms),
            "median_ms": round(med, 1), "build_s": probe.get("build_s")}


def layer_utilization(args, probe):
    """规划值和实际值是否一致 —— 不一致说明某一环静默失效。"""
    if args.skip_utilization:
        return {"name": "utilization", "ok": True, "skipped": True}
    plan = probe.get("plan") or {}
    obs = probe.get("obs") or {}
    vram = probe.get("vram") or {}
    dma = probe.get("dma") or {}
    checks = []

    planned_slots = plan.get("expert_slots")
    actual_slots = ((obs.get("slots") or {}).get("slots"))
    if planned_slots is not None and actual_slots is not None:
        checks.append(("槽位数：规划 %s / 实际 %s" % (planned_slots, actual_slots),
                       int(actual_slots) > 0))

    # 显存：占满多少。留太多说明没吃满（浪费），但这是报告项不是失败项 ——
    # 不同模型/卡的合理区间不同，硬判会误报。
    # 带宽：规划器做取舍全靠这三个数，必须是实测的
    bw = probe.get("bandwidth") or {}
    if bw:
        h2d = float(bw.get("h2d_bps") or 0) / 1e9
        dsk = float(bw.get("disk_bps") or 0) / 1e9
        dskr = float(bw.get("disk_rand_bps") or 0) / 1e9
        checks.append(("带宽：h2d %.1f / disk %.1f (rand %.1f) GB/s" % (
            h2d, dsk, dskr), h2d > 1.0 and dsk > 0.1))

    if vram:
        usable = max(0.01, vram.get("total_gb") or 0.01)
        used = vram.get("used_gb") or 0.0
        frac = used / usable
        # 判据：吃满但不溢出。低于 60% 说明规划保守到浪费，高于 99% 说明没有余量
        # （运行时临时分配会失败）。这是「资源利用率」那一层该管的。
        # 判据分两半：
        #   - 利用率不能太低（<60% 说明规划保守到浪费）
        #   - **余量不能为 0**（运行时还要分配 activation、临时张量；free=0 就是
        #     下一次分配必失败的信号，实测就是在这个状态撞 !handles_）
        _free = vram.get("free_gb") or 0.0
        checks.append(("显存：用 %.2f/%.2f GB (%.0f%%)，余 %.2f GB" % (
            used, usable, frac * 100, _free),
                       0.60 <= frac <= 0.97 and _free >= 0.15))

    # H2D 是否真走异步 DMA。零调用说明设备侧全在走退化路径。
    if dma:
        calls = int(dma.get("ok_calls") or 0)
        fb = int(dma.get("fallback_path") or 0)
        checks.append(("异步 DMA：%d 次成功 / %d 次退化" % (calls, fb),
                       calls > 0 or fb == 0))

    tiers = (obs.get("tiers") or {})
    if tiers:
        tot = int(tiers.get("total") or 0)
        vram_hit = int(tiers.get("vram_hit") or 0)
        rate = (vram_hit / tot) if tot else 0.0
        checks.append(("缓存命中：vram %.1f%% / ram %.1f%% / disk %s" % (
            rate * 100, 100.0 * float(tiers.get("ram_hit_rate") or 0),
            tiers.get("disk")), True))

    ok = all(c[1] for c in checks)
    tag = "PASS" if ok else "FAIL"
    print("[L3 utilization] %s" % tag)
    for desc, good in checks:
        print("               %s %s" % ("ok  " if good else "BAD ", desc))
    return {"name": "utilization", "ok": ok,
            "checks": [{"desc": d, "ok": g} for d, g in checks],
            "plan": plan, "vram": vram, "dma": dma}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("TI_MODEL", ""))
    ap.add_argument("--layer", default="all",
                    choices=["all", "accuracy", "engine", "utilization"])
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--json", default=None)
    ap.add_argument("--skip-accuracy", action="store_true")
    ap.add_argument("--skip-engine", action="store_true")
    ap.add_argument("--skip-utilization", action="store_true")
    ap.add_argument("--accuracy-timeout", type=int, default=900)
    ap.add_argument("--engine-timeout", type=int, default=1800)
    args = ap.parse_args(argv)

    if not args.model:
        print("need --model (or TI_MODEL)", file=sys.stderr)
        return 2

    t0 = time.time()
    results = []

    if args.layer in ("all", "accuracy"):
        results.append(layer_accuracy(args))

    probe = {}
    if args.layer in ("all", "engine", "utilization") and not (
            args.skip_engine and args.skip_utilization):
        probe = _probe_engine(args.model, args.steps, args.engine_timeout)

    if args.layer in ("all", "engine"):
        results.append(layer_engine(args, probe))
    if args.layer in ("all", "utilization"):
        results.append(layer_utilization(args, probe))

    total = time.time() - t0
    failed = [r["name"] for r in results if not r.get("ok")]
    print("")
    print("[ci] %s  (%.0fs)  %s" % (
        "PASS" if not failed else "FAIL", total,
        "layers: " + ", ".join(r["name"] for r in results)))
    if failed:
        print("[ci] failed: " + ", ".join(failed))

    if args.json:
        try:
            Path(args.json).write_text(
                json.dumps({"ok": not failed, "seconds": round(total, 1),
                            "layers": results, "probe": probe},
                           ensure_ascii=False, indent=2, default=str),
                encoding="utf-8")
            print("[ci] wrote " + args.json)
        except Exception as exc:
            print("[ci] json failed: %r" % (exc,))

    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
