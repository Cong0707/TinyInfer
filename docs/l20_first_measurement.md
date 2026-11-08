# L20 首次实测：40/40 层全驻留，但数字远低于预期（2026-11-08 15:27）

## 规划器在 L20 上的判断（好）

```
PLAN NVIDIA L20 vram=44.1GB -> layers=40/40 stream=0 buf=0
     slots=1057 chunk=132 embed=gpu head=gpu
     cache=283.8GB
```

- **40 层全部常驻**，无需 streamed 层
- **1057 个专家槽位**，覆盖一个 decode step 的 4.4 倍（240 需求）
- 启动时明确打印 `expert slots cover a full decode step (1057/240); steady-state fetch is zero`
- RAM 缓存 283.8GB，能装下全部 15360 个专家（289GB）

这与之前的推算完全吻合。

## 但实测数字远低于预期

```
BENCH rows=1  1046.7 ms/step   0.96 tok/s   finite=True
BENCH rows=2  1134.8 ms/step   1.76 tok/s   finite=False
BENCH rows=4   151.6 ms/step  26.39 tok/s   finite=False
BENCH rows=8   774.1 ms/step  10.34 tok/s   finite=False
```

预算是「40 层 x 0.95ms(MoE) + attention + hc ≈ 64ms/step ≈ 15 tok/s」，
实测 1046ms，**慢 16 倍**。

## 两个独立的严重问题

### 1. `cudart not found` 让 async 流水线完全失效

```
[async] expert pipeline unavailable (RuntimeError('cudart not found 
       (needed for host-mapped memory)')); falling back to the per-layer sync path
(刷了 20 行)
```

**根因**：`budget.py` 的 `_cudart()` 只 glob `torch/lib/cudart*.dll`（Windows 名字）。
而 torch 2.15 的 Linux wheel **不带** libcudart（静态链接了运行时），
真正的 so 在 `site-packages/nvidia/cu13/lib/` 或 `/usr/local/cuda/lib64/`。

**后果**：GPU 主动下单、CPU 立刻推送这条设计在 L20 上**根本没跑起来**，
退化成了逐层同步 —— 那正是慢 16 倍最可能的原因。

**已修**（两轮）：先加 `libcudart*.so*`，再加 `nvidia/cu13/lib`、`nvidia/cuda_runtime/lib`
和裸 soname。修完后 `cudart: True`，`host_mapped: OK`。

### 2. rows>1 全部 NaN（本机 8GB 上正常）

```
rows=1 finite=True
rows=2 finite=False
rows=4 finite=False
rows=8 finite=False
```

**这是 L20 特有的**：同一份代码在本机 8GB 卡上 rows=2 是 finite 的。

差异在于 L20 上 `embed=gpu head=gpu`（本机是 `embed=cpu head=gpu`），
以及 40 层全驻留（本机 11 层）。

## 下一步

1. 等 async 修好后的基准（正在跑）
2. 如果 async 生效后 NaN 仍在，说明是 embed/head 在 GPU 上的 batching 问题
3. 修好 NaN 之后再谈吞吐
