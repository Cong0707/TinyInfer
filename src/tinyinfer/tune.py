"""训练器：校准规划器的实测输入。

## 这里没有「基线」，也没有「坐标下降」

我一开始把训练器写成了「找一个比基线更快的配置」，那是错的 —— 那种形态默认
**存在一个需要人设定的参照状态**。而现在布局由目标函数从资源量算出来，只有
一个解，不需要和谁比。

## 训练器真正该做的

规划器的输入分三类：

  1. 模型几何   从 checkpoint header 读（层/专家/embed/head 字节）—— 确定
  2. 硬件容量   从驱动读（显存/内存）                          —— 确定
  3. 硬件带宽   只能实测                                      —— **可能不准**

前两类没有「调」的余地。第三类是**唯一的校准对象**：一次测量可能被页缓存、
别的进程、冷启动影响，测出来偏了，规划器就会按错的带宽解出错的布局。

所以训练器做三件事：

  calibrate()  把带宽测多次，取稳健值（不是取一次，也不是取最快）
  verify()     用校准后的带宽让规划器解一次，再跑真实引擎验证
  commit()     写回 tuning.json（source=measured），下次启动直接用

## 「稳健值」怎么取

带宽测量有三类噪声，各用不同办法对付：

  冷启动     第一次搬会触发页错误/建 context —— 丢弃前 N 次
  页缓存     读过的数据第二次会命中内存 —— 用随机偏移，避免重读同一块
  瞬时抖动   别的进程抢 IO —— 取多次的中位数，不是均值

实测这台机器：单次测 H2D 得 7.1GB/s，稳定；但 disk 单次能得 8.7GB/s（页缓存
命中）也能得 4.4（真读盘），差一倍。**不区分这两种，规划器就会按 8.7 去解。**

## 用法

    python -m tinyinfer.tune --model <checkpoint>          # 校准并验证
    python -m tinyinfer.tune --model <checkpoint> --dry-run  # 只看不写
    python -m tinyinfer.tune --model <checkpoint> --show     # 打印当前实测值
"""
from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

NL = chr(10)
ROOT = Path(__file__).resolve().parents[1]

# 带宽采样：多次取中位数。少于这个数，一个异常样本就能把结果带偏。
_BW_RUNS = 3


def _src_dir() -> str:
    return str(ROOT / "src")


# ---- 带宽校准 --------------------------------------------------------------

def calibrate_bandwidth(model_dir, runs=_BW_RUNS, verbose=True):
    """多次实测带宽，取每项的中位数。

    **不能只测一次**：disk 单次可能撞上页缓存（8.7GB/s）也可能是真读盘
    （4.4GB/s），差一倍。规划器按哪个数解，得出的布局完全不同。

    也不能取最快值 —— 最快的那个几乎总是页缓存命中，不代表真实读盘。
    取中位数，让异常样本互相抵消。
    """
    from tinyinfer.resources import detect

    hw = detect()
    samples = {"h2d_bps": [], "disk_bps": [], "ram_bps": []}
    for i in range(max(1, int(runs))):
        # 每次 force=True 跳过缓存，真的重新测
        bw = hw.bandwidth({}, model_dir=str(model_dir))
        for k in samples:
            v = bw.get(k)
            if v:
                samples[k].append(float(v))
        if verbose:
            print("[tune] 第 %d 次: h2d=%.1f disk=%.1f ram=%.1f GB/s" % (
                i + 1, (bw.get("h2d_bps") or 0) / 1e9,
                (bw.get("disk_bps") or 0) / 1e9,
                (bw.get("ram_bps") or 0) / 1e9), flush=True)

    out = {}
    for k, vs in samples.items():
        if not vs:
            continue
        out[k] = float(statistics.median(vs))
        if verbose and len(vs) > 1:
            spread = (max(vs) - min(vs)) / max(1.0, out[k])
            print("[tune] %-10s 中位数 %.1fGB/s（极差 %.0f%%）" % (
                k, out[k] / 1e9, spread * 100), flush=True)
    return out


# ---- 验证：让规划器解一次，再用真引擎跑 --------------------------------

_PROBE = """
import json, sys, time, statistics
sys.path.insert(0, __SRC__)
import torch
from tinyinfer.engine import Engine
from tinyinfer.observability import collect_observability

M = __MODEL__
WARM_ROUNDS = __WARMR__
MEASURE_ROUNDS = __MEASR__

out = {"error": None, "samples": [], "layout": None, "correct": None}
try:
    eng = Engine.from_hf(M, device="cuda", allow_missing_layers=True)
    ids = torch.tensor([[100, 200, 300, 400, 500]], dtype=torch.long, device="cuda")
    with torch.inference_mode():
        eng.forward(ids, 0)
        torch.cuda.synchronize()

        # 预热到稳定：连续 3 步读盘量变化 < 10%。
        # 不预热测的是「后台 warm 线程跑到哪了」，不是布局好坏。
        rd = getattr(getattr(eng, "expert_store", None), "reader", None)
        prev, stable = None, 0
        for _ in range(80):
            b0 = getattr(rd, "bytes_read", 0)
            eng.forward(ids, 0)
            torch.cuda.synchronize()
            gb = (getattr(rd, "bytes_read", 0) - b0) / 1e9
            if prev is not None and prev > 0 and abs(gb - prev) < prev * 0.10:
                stable += 1
                if stable >= 3:
                    break
            else:
                stable = 0
            prev = gb

        for _ in range(MEASURE_ROUNDS):
            if hasattr(eng, "reset"):
                eng.reset()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            eng.forward(ids, 0)
            torch.cuda.synchronize()
            out["samples"].append((time.perf_counter() - t0) * 1e3)

        if hasattr(eng, "reset"):
            eng.reset()
        torch.cuda.synchronize()
        a = eng.forward(ids, 0)
        torch.cuda.synchronize()
        if hasattr(eng, "reset"):
            eng.reset()
        torch.cuda.synchronize()
        b = eng.forward(ids, 0)
        torch.cuda.synchronize()
        fin = bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all())
        if fin:
            d = float((a.float() - b.float()).abs().max())
            sc = float(a.float().abs().max()) + 1e-6
            out["correct"] = {"finite": True, "repeatable": d <= 1e-2 * sc}
        else:
            out["correct"] = {"finite": False, "repeatable": False}

    p = getattr(eng, "hw_plan", None)
    if p is not None:
        out["layout"] = {k: getattr(p, k, None) for k in (
            "resident", "slots", "stream_layers")}
    obs = collect_observability(eng)
    out["tiers"] = obs.get("tiers")
    try:
        eng.close()
    except Exception:
        pass
except Exception as exc:
    out["error"] = repr(exc)[:300]
print("TUNEJSON" + json.dumps(out))
"""


def verify_layout(model_dir, *, warm_rounds=1, measure_rounds=6, timeout=1800):
    """让引擎按当前规划跑一次，检查布局是否真的能用。

    这一步是必须的：目标函数是**估算**，它不知道真实的 kernel workspace、
    分配器碎片、KV 实际大小。跑一次真实引擎才知道解出来的布局能不能落地。
    """
    code = (_PROBE
            .replace("__SRC__", repr(_src_dir()))
            .replace("__MODEL__", repr(str(model_dir)))
            .replace("__WARMR__", str(int(warm_rounds)))
            .replace("__MEASR__", str(int(measure_rounds))))
    e = os.environ.copy()
    e.setdefault("PYTHONPATH", _src_dir())
    try:
        p = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), env=e,
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"error": "timeout"}
    for line in (p.stdout or "").splitlines():
        if line.startswith("TUNEJSON"):
            try:
                raw = json.loads(line[8:])
            except Exception:
                return {"error": "bad json"}
            s = sorted(raw.get("samples") or [])
            if s:
                raw["median_ms"] = round(statistics.median(s), 1)
                raw["spread"] = round((s[-1] - s[0]) / max(1.0, s[-1]), 3)
            return raw
    tail = (p.stderr or "").strip().splitlines()
    return {"error": (tail[-1] if tail else "no output")[:200]}


def judge(result):
    """一份验证结果是否可用。返回 (ok, 原因)。

    顺序是刻意的：**正确性不过关的布局，它的时间数字没有意义。**
    """
    if not result or result.get("error"):
        return False, "失败: %s" % (result or {}).get("error")
    c = result.get("correct") or {}
    if not c.get("finite"):
        return False, "输出含 NaN/Inf"
    if not c.get("repeatable"):
        return False, "同输入两次不一致"
    if (result.get("spread") or 0) > 0.40:
        return False, "测量不稳（极差 %.0f%%）" % (100 * result.get("spread", 0))
    return True, "ok"


# ---- 主流程 ----------------------------------------------------------------

def run(model_dir, *, dry_run=False, verbose=True, runs=_BW_RUNS):
    """校准 -> 验证 -> 写回。没有「基线」，没有坐标下降。"""
    from tinyinfer import tuning

    model_dir = str(model_dir)

    # 1. 校准带宽（唯一需要实测的输入）
    if verbose:
        print("[tune] 校准带宽（测 %d 次取中位数）" % runs, flush=True)
    bw = calibrate_bandwidth(model_dir, runs=runs, verbose=verbose)
    if not bw:
        return {"error": "带宽测不出来"}

    # 2. 写进配置，让规划器按它解（先写，验证才用得上）
    if not dry_run:
        tuning.update(model_dir, {k: float(v) for k, v in bw.items()},
                      source="measured")
        if verbose:
            print("[tune] 已写入 tuning.json", flush=True)

    # 3. 用这套带宽跑真实引擎，验证解出来的布局能落地
    if verbose:
        print("[tune] 验证布局", flush=True)
    res = verify_layout(model_dir)
    ok, why = judge(res)
    if verbose:
        lay = res.get("layout") or {}
        print("[tune] %s  %s  %s" % (
            "PASS" if ok else "FAIL",
            "resident=%s slots=%s" % (lay.get("resident"), lay.get("slots")),
            ("%.0fms (极差 %.0f%%)" % (res.get("median_ms", 0),
                                       100 * res.get("spread", 0))) if ok else why),
            flush=True)

    out = {"bandwidth": bw, "verified": res.get("layout"),
           "ok": ok, "note": why}
    if not ok and not dry_run:
        # 验证失败不该留下「measured」标记 —— 那会让下次启动盲目信任它
        tuning.update(model_dir, {"last_verify_failed": True}, source="runtime")
    return out


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--show", action="store_true", help="打印当前配置里存的实测值")
    ap.add_argument("--runs", type=int, default=_BW_RUNS)
    a = ap.parse_args(argv)

    if a.show:
        from tinyinfer import tuning

        print(json.dumps(tuning.params(a.model), ensure_ascii=False, indent=2))
        return 0

    res = run(a.model, dry_run=a.dry_run, runs=max(1, a.runs))
    print("")
    print(json.dumps(res, ensure_ascii=False, indent=2)[:2000])
    return 0 if res.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
