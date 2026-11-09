"""观测数据的唯一出口。

引擎里能观测的东西散落在四个对象上：

  Engine            —— 规划结果（驻留层数、槽位数、分块）、设备、tier
  ExpertStore       —— 三级存储的字节账（显存槽位 / 内存缓存 / 磁盘读取）
  VramSlotCache     —— 槽位池的实时状态（各状态格数、占用、最忙的层）
  ExpertTiers       —— 按专家数计的命中（vram_hit / ram_hit / disk）

每个都有自己的 stats()，字段名和口径各写各的，于是出现两个必须避免的问题：

1. **同一件事有两个数。** 「专家命中率」一度在 VramSlotCache.hits（按 acquire
   调用次数）和 ExpertTiers.stats["vram_hit"]（按专家个数）各算一遍，批量路径下
   一次 acquire_batch 只 +1 却命中多个专家，两个数永远对不上。
2. **调用方各取所需。** /health 取一部分、bench 取一部分、日志又取一部分，
   同一个指标在不同地方显示不同值。

这个模块只做一件事：把上述四处的统计收成一个固定形状的 dict，所有出口
（/health、bench、日志）都从这里取。字段名和口径在这里定义一次。

**口径约定**

- 命中数一律按「专家个数」计，不按「调用次数」计。
- 字节数一律用十进制 GB（1e9），与厂商口径一致；字节本身用 int。
- 派生量（命中率、覆盖率）在**能算的时候**才给，算不出就是 None，
  不要用 0 冒充 —— 0 和「不知道」在排查时是两件事。
"""
from __future__ import annotations

import time
from typing import Any


NL = chr(10)

def _safe(fn, default=None):
    """调一个可能不存在的统计方法；失败一律返回 default，不让观测拖垮服务。"""
    if not callable(fn):
        return default
    try:
        return fn()
    except Exception:
        return default


def collect_observability(engine, *, include_slots: bool = True,
                          include_tiers: bool = True) -> dict[str, Any]:
    """把引擎当前的观测状态收成一个 dict。

    engine 可以是 Engine，也可以是 None（服务未加载模型时）。任何一项取不到
    就填 None，**绝不抛异常** —— 观测代码不能成为新的故障源。
    """
    out: dict[str, Any] = {"ts": time.time(), "engine": None, "slots": None,
                           "tiers": None, "store": None}

    if engine is None:
        return out

    # ---- 规划结果与设备 ------------------------------------------------
    plan = getattr(engine, "hw_plan", None)
    layers = getattr(engine, "max_layers", None)
    resident = getattr(engine, "gpu_layers", None)
    streamed = None
    try:
        streamed = len(engine._stream_layers())
    except Exception:
        streamed = (layers - resident) if (layers and resident is not None) else None

    out["engine"] = {
        "device": str(getattr(engine, "device", "?")),
        "tier": getattr(getattr(engine, "tier", None), "level", None),
        "layers_total": layers,
        "layers_resident": resident,
        "layers_streamed": streamed,
        # 这三个是规划器解出来的，不是常量：换机器就变
        "expert_slots_planned": getattr(plan, "expert_slots", None),
        "prefill_chunk": getattr(plan, "prefill_chunk", None),
        "host_cache_gb": (round(plan.cache_bytes / 1e9, 2)
                          if getattr(plan, "cache_bytes", None) else None),
        "head_on_cpu": getattr(plan, "head_on_cpu", None),
        "embed_on_cpu": getattr(plan, "embed_on_cpu", None),
        # decode 图是否已建（没建说明还在启动期或捕获失败）
        "cuda_graph": getattr(engine, "_graph", None) is not None,
    }

    store = getattr(engine, "expert_store", None)

    # ---- 槽位池 --------------------------------------------------------
    # 注意：这里**不算命中率**。槽位池只知道「搬了多少次」，不知道「有多少个
    # 专家本来就够用」——后者要按专家数数，只有 ExpertTiers.need() 拿得到。
    if include_slots and store is not None:
        slots = getattr(store, "slots", None)
        snap = _safe(getattr(slots, "snapshot", None))
        if snap is not None:
            out["slots"] = snap

    # ---- 按专家数的逐级命中 --------------------------------------------
    if include_tiers and store is not None:
        # 分级管理器由引擎创建、挂在 store 的 _tier_mgr 上（全机只有一个）。
        # 它是唯一按「专家个数」计数的口径，命中率只能从这里算。
        tiers = getattr(store, "_tier_mgr", None) or getattr(store, "tiers", None)
        if tiers is not None:
            st = getattr(tiers, "stats", None)
            if isinstance(st, dict):
                out["tiers"] = dict(st)
                # 命中率在这里算：这是唯一按专家个数计数的口径
                vram = int(st.get("vram_hit", 0) or 0)
                ram = int(st.get("ram_hit", 0) or 0)
                disk = int(st.get("disk", 0) or 0)
                total = vram + ram + disk
                out["tiers"]["total"] = total
                out["tiers"]["vram_hit_rate"] = (vram / total) if total else None
                out["tiers"]["ram_hit_rate"] = (ram / total) if total else None

    # ---- 存储侧的字节账 ------------------------------------------------
    if store is not None:
        raw = _safe(getattr(store, "stats", None))
        if isinstance(raw, dict):
            out["store"] = raw

    return out


def format_summary(obs: dict[str, Any]) -> str:
    """把观测结果压成几行给人看。用于启动日志和 bench 结尾。"""
    e = obs.get("engine") or {}
    t = obs.get("tiers") or {}
    s = obs.get("slots") or {}
    lines = []
    if e:
        lines.append(
            f"层 {e.get('layers_resident')}/{e.get('layers_total')}"
            f"（streamed {e.get('layers_streamed')}）"
            f"  槽位 {e.get('expert_slots_planned')}"
            f"  chunk {e.get('prefill_chunk')}"
            f"  缓存 {e.get('host_cache_gb')}GB"
            f"  graph {'on' if e.get('cuda_graph') else 'off'}")
    if t:
        rate = t.get("vram_hit_rate")
        lines.append(
            f"专家命中 vram={t.get('vram_hit', 0)} ram={t.get('ram_hit', 0)}"
            f" disk={t.get('disk', 0)}"
            + (f"  显存命中率 {rate * 100:.1f}%" if rate is not None else "  显存命中率 n/a"))
    if s:
        states = s.get("states") or {}
        lines.append(
            f"槽位占用 {s.get('occupied')}/{s.get('slots')}"
            f"  in_use={states.get('in_use', 0)} staged={states.get('staged', 0)}"
            f"  loads={s.get('loads')} evictions={s.get('evictions')}")
    return "\n".join(lines)


# ---- 统一调试入口 -----------------------------------------------------------
#
# 以前排查一个问题要开好几个开关、跑好几次、把几次的输出拼起来看：
#
#   TI_MEM_TRACE=1   一次（看显存）
#   TI_MOE_PROF=1    又一次（看 MoE 各段耗时）
#   TI_CACHE_TRACE=1 又一次（看缓存命中）
#   TI_TIME=1        又一次（看时间分布）
#   TI_DEBUG_NAN=all 又一次（看是否 NaN）
#
# 问题不只是费时间：**每次运行的环境都不一样**（预热进度、页缓存温度、显存碎片），
# 拼起来的"完整证据"其实来自几个不同的状态，互相之间不可比。有一次就是因为这个
# 把「连续 4 次干净」当成了找到根因，复跑才发现是 6% 概率的巧合。
#
# 所以调试要一次拿全：`TI_DIAG=1` 打开，一次 forward 之后把下面这些一起 dump。
# 判据固定、采样固定、输出固定，多次之间的差异就只来自被测对象本身。


def diag_enabled() -> bool:
    """唯一的调试开关。所有诊断数据都走它，不再有十来个独立开关。"""
    import os

    return os.environ.get("TI_DIAG") not in (None, "", "0")


def collect_diag(engine, *, samples=None, forward_ms=None) -> dict[str, Any]:
    """一次采集完整的调试证据。

    与 collect_observability 的分工：那个给运维看（稳定、字段少、可以放心读）；
    这个给排查看（要全、可以重、允许包含大块原始数据）。
    """
    import torch

    out: dict[str, Any] = {
        "ts": time.time(),
        "hardware": None,
        "plan": None,
        "memory": None,
        "transfer": None,
        "cache": None,
        "compute": None,
        "correctness": None,
        "timing": None,
    }

    try:
        from . import tuning

        out["hardware"] = {"key": tuning.hardware_key(),
                           "params": tuning.params(getattr(engine, "_diag_model_dir", ""))}
    except Exception:
        pass

    plan = getattr(engine, "hw_plan", None)
    if plan is not None:
        out["plan"] = {k: getattr(plan, k, None) for k in (
            "expert_slots", "cache_bytes", "host_pool_bytes", "pin_ratio",
            "read_threads", "gpu_layers", "stream_layers", "prefill_chunk",
            "head_on_cpu", "embed_on_cpu", "fits", "reason")}

    # 显存：已分配/保留/峰值/碎片。碎片高说明分配器在反复找块。
    try:
        if torch.cuda.is_available():
            out["memory"] = {
                "allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 3),
                "reserved_gb": round(torch.cuda.memory_reserved() / 1e9, 3),
                "max_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
                "free_gb": round(
                    (torch.cuda.get_device_properties(0).total_memory
                     - torch.cuda.memory_reserved()) / 1e9, 3),
                "retries": int(getattr(
                    torch.cuda.memory_stats().get("num_alloc_retries", 0), 0)),
            }
    except Exception:
        pass

    # 传输：异步 DMA 是否真的走了、退化了几次
    try:
        from .loading import _async_h2d_stats

        out["transfer"] = _async_h2d_stats()
    except Exception:
        pass

    # 缓存与命中（从 collect_observability 拿，口径唯一）
    try:
        obs = collect_observability(engine)
        out["cache"] = {"slots": obs.get("slots"), "tiers": obs.get("tiers")}
    except Exception:
        pass

    # 计算：MoE 各段累计耗时（原来只有 TI_MOE_PROF=1 才收）
    try:
        agg: dict[str, float] = {}
        for lid in range(int(getattr(engine, "max_layers", 0) or 0)):
            blk = _safe(lambda i=lid: engine._blocks.get(i))
            ffn = getattr(blk, "ffn", None) if blk is not None else None
            parts = getattr(ffn, "_parts", None)
            if parts:
                for k, v in parts.items():
                    agg[k] = agg.get(k, 0.0) + float(v)
        if agg:
            out["compute"] = {k: round(v * 1e3, 1) for k, v in sorted(agg.items())}
    except Exception:
        pass

    # 正确性：输出是否有限、同输入是否可重复（这条比速度重要）
    if samples is not None:
        out["correctness"] = samples

    if forward_ms is not None:
        out["timing"] = {"forward_ms": forward_ms}

    return out


def format_diag(diag: dict[str, Any]) -> str:
    """把 collect_diag 的结果排成可读的报告。"""
    lines = ["[diag] " + time.strftime("%H:%M:%S")]
    hw = diag.get("hardware") or {}
    if hw:
        lines.append("  hardware : " + str(hw.get("key")))
        p = hw.get("params") or {}
        if p:
            lines.append("  params   : " + ", ".join("%s=%s" % (k, v) for k, v in p.items()))
    plan = diag.get("plan") or {}
    if plan:
        lines.append("  plan     : slots=%s cache=%sGB pool=%sGB pin=%s readers=%s layers=%s/%s" % (
            plan.get("expert_slots"),
            round((plan.get("cache_bytes") or 0) / 1e9, 2),
            round((plan.get("host_pool_bytes") or 0) / 1e9, 2),
            plan.get("pin_ratio"), plan.get("read_threads"),
            plan.get("gpu_layers"),
            (plan.get("gpu_layers") or 0) + (plan.get("stream_layers") or 0)))
    mem = diag.get("memory") or {}
    if mem:
        lines.append("  memory   : alloc=%.2fG reserved=%.2fG peak=%.2fG free=%.2fG retries=%d" % (
            mem.get("allocated_gb", 0), mem.get("reserved_gb", 0),
            mem.get("max_allocated_gb", 0), mem.get("free_gb", 0),
            mem.get("retries", 0)))
    tr = diag.get("transfer") or {}
    if tr:
        lines.append("  transfer : async=%s fallback=%s rcfail=%s bytes=%.2fGB" % (
            tr.get("ok_calls"), tr.get("fallback_path"), tr.get("rc_fail"),
            (tr.get("ok_bytes") or 0) / 1e9))
    tiers = (diag.get("cache") or {}).get("tiers") or {}
    if tiers:
        lines.append("  cache    : vram_hit=%.1f%% ram_hit=%.1f%% disk=%s" % (
            100.0 * (tiers.get("vram_hit_rate") or 0),
            100.0 * (tiers.get("ram_hit_rate") or 0), tiers.get("disk")))
    comp = diag.get("compute") or {}
    if comp:
        lines.append("  compute  : " + "  ".join("%s=%.0fms" % (k, v)
                                                for k, v in list(comp.items())[:8]))
    corr = diag.get("correctness")
    if corr:
        lines.append("  correct  : " + str(corr))
    tm = diag.get("timing") or {}
    if tm:
        lines.append("  timing   : %.0fms/step" % tm.get("forward_ms", 0))
    return NL.join(lines)
