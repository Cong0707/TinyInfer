"""训练器：实测扫参，把最优配置固化下来给冷启动用。

引擎里曾经写死一堆「在这台 47GB 笔记本上调出来的」常数：宿主池留 6GB、
pin 比例 0.5、读线程 16、slab 系数 1.15。换一台机器就静默地不对，
表现为「明明有 300GB 只用 32GB」。这些数不该由人拍，应该由训练器实测。

分工：
  calibrate.py  启动期轻量测量（磁盘带宽、读并发），秒级，每次启动都跑（有缓存）
  tune.py       离线扫参，分钟级，产出这台机器的最优参数，写进 tuned.json

冷启动三级：tuned.json 训练结果 > calibration.json 本机缓存 > 现场实测。

用法：
    python -m tinyinfer.tune --model DIR
    python -m tinyinfer.tune --model DIR --dry-run
    python -m tinyinfer.tune --list
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

TUNED_PATH = Path(__file__).with_name("tuned.json")
# 换行符用 chr(10)：这类文件的历史改动里出现过换行转义被工具的字符串处理吃掉、
# 变成真实换行把字符串撑断的情况。显式 chr(10) 不会有这个问题。
NL = chr(10)


def machine_key() -> str:
    """机器指纹：只取影响性能的硬指标。"""
    bits = []
    try:
        import torch

        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            bits.append("gpu=" + torch.cuda.get_device_name(0).replace(" ", "_"))
            bits.append("vram=%dG" % int(p.total_memory // (1 << 30)))
    except Exception:
        bits.append("gpu=none")
    try:
        import psutil

        bits.append("ram=%dG" % int(psutil.virtual_memory().total // (1 << 30)))
    except Exception:
        pass
    bits.append("os=" + sys.platform)
    return "|".join(bits)


def load_tuned() -> dict:
    try:
        return json.loads(TUNED_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"version": 1, "entries": {}}


def save_tuned(data: dict) -> None:
    TUNED_PATH.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + NL, encoding="utf-8")


def lookup(model_dir=None) -> dict | None:
    """冷启动入口：这台机器有没有训练过的配置。

    参数描述的是机器（哪个盘快、几个并发最优），不是权重，所以同一台机器上
    换模型仍然适用 —— 不按模型名过滤。
    """
    data = load_tuned()
    entry = (data.get("entries") or {}).get(machine_key()) or {}
    return entry.get("chosen") or None


def _bench_step(eng, pos: int, n: int = 3):
    """跑 n 步 decode，返回 (每步毫秒, 每步读盘 GB)。"""
    import torch

    r0 = eng.expert_store.reader
    b0 = r0.bytes_read
    t0 = time.perf_counter()
    with torch.inference_mode():
        for i in range(n):
            eng.forward(torch.tensor([[900 + i]], device="cuda"), pos + i)
    torch.cuda.synchronize()
    el = (time.perf_counter() - t0) / n
    return el * 1e3, (r0.bytes_read - b0) / n / 1e9


def tune(model_dir, *, dry_run: bool = False) -> dict:
    """扫参并记录。"""
    import torch

    from .engine import Engine

    eng = Engine.from_hf(str(model_dir), device="cuda", allow_missing_layers=True)
    recorded = {
        "key": machine_key(),
        "model": Path(str(model_dir)).name,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "measured": {},
        "chosen": {},
    }
    try:
        ram = eng.expert_store.ram
        with torch.inference_mode():
            eng.forward(torch.tensor([[1, 2, 3]], device="cuda"), 0)
        pos = 10

        # **先预热到稳定，再开始比较。**
        #
        # 这一步不能省。后台 warm 线程一直在往缓存里搬专家，所以「刚加载完」和
        # 「跑了 20 步之后」的每步读盘量能差 3 倍（实测 10.89GB -> 3.27GB）。
        # 如果不预热就直接扫参，测出来的差异全是「此刻预热跑到哪了」，
        # 跟被扫的参数没有关系 —— 第一版就是这么被骗的：pin 从 0.3 到 1.0
        # 看着像 2880->2696->2742 的曲线，实际那只是时间在走。
        #
        # 稳定判据：连续 3 步的读盘量变化都小于 10%。
        _prev = None
        _stable = 0
        for _ in range(60):
            _ms, _gb = _bench_step(eng, pos, 1)
            pos += 1
            if _prev is not None and _prev > 0 and abs(_gb - _prev) < _prev * 0.10:
                _stable += 1
                if _stable >= 3:
                    break
            else:
                _stable = 0
            _prev = _gb
        print("[tune] warmed up after %d steps" % pos, flush=True)

        base_ms, base_gb = _bench_step(eng, pos)
        print("[tune] baseline step=%.0fms read=%.2fGB" % (base_ms, base_gb), flush=True)
        recorded["measured"]["baseline"] = {
            "step_ms": round(base_ms, 1), "read_gb": round(base_gb, 3)}

        best_pin, best_ms = None, base_ms
        cap = int(ram.capacity_entries * ram.blob_bytes)
        for ratio in (0.3, 0.5, 0.7, 1.0):
            try:
                ram.pin_budget = int(cap * ratio)
                ram._pinned_bytes = 0
                pos += 10
                ms, gb = _bench_step(eng, pos)
                recorded["measured"]["pin_%.1f" % ratio] = {
                    "step_ms": round(ms, 1), "read_gb": round(gb, 3)}
                print("[tune] pin=%.1f step=%.0fms read=%.2fGB" % (ratio, ms, gb),
                      flush=True)
                if ms < best_ms:
                    best_pin, best_ms = ratio, ms
            except Exception as exc:
                print("[tune] pin=%.1f failed: %r" % (ratio, exc), flush=True)
        recorded["chosen"]["pin_ratio"] = best_pin if best_pin is not None else 0.5
        recorded["chosen"]["read_threads"] = int(
            getattr(eng.expert_store.reader, "threads", 0) or 0) or None
        recorded["measured"]["best_step_ms"] = round(best_ms, 1)
        print("[tune] chosen pin_ratio=%s read_threads=%s step=%.0fms"
              % (recorded["chosen"]["pin_ratio"],
                 recorded["chosen"]["read_threads"], best_ms), flush=True)
    finally:
        try:
            eng.close()
        except Exception:
            pass

    if not dry_run:
        data = load_tuned()
        data.setdefault("entries", {})[recorded["key"]] = recorded
        save_tuned(data)
        print("[tune] wrote " + str(TUNED_PATH), flush=True)
    return recorded


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args(argv)
    if args.list:
        ents = (load_tuned().get("entries") or {})
        if not ents:
            print("(no entries)")
        for k, v in ents.items():
            ch = v.get("chosen") or {}
            best = (v.get("measured") or {}).get("best_step_ms")
            print("%s  model=%s ts=%s pin=%s threads=%s step=%sms"
                  % (k, v.get("model"), v.get("ts"), ch.get("pin_ratio"),
                     ch.get("read_threads"), best))
        return
    if not args.model:
        ap.error("--model is required unless --list")
    tune(args.model, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
