# L20 + CUDA graph：21.2 ms/step，47.2 tok/s

日期 2026-11-08 17:29

## 数字

```
all steps: 11452 20 19 19 19 21 28 19 19 19 22 21   (ms)
steady 21.2ms  47.16 tok/s
```

首步 11452ms 是捕获（10 个 chunk，每 chunk 4 层），之后稳定 19-22ms。

## 之前「graph 跑不通」是个假象

我一直用 `_forward_chunk()` 测，那是内部实现，**绕开了 `forward()` 里的整个
graph 分支**。真正的入口是 `forward()`：

```python
hit = self._graph_run(ids, start_pos)
if hit is not None: return hit
self._graph_try_capture(ids, start_pos)
hit = self._graph_run(ids, start_pos)
```

换到 `forward()` 后第一次就捕获成功：

```
[tinyinfer] decode CUDA graph captured at pos=128 (10 chunks)
graph=True finite=True
```

## 同时做的两件事

1. **删掉 eager 回落。** `forward()` 原来三元分支（graph 命中 / 捕获 / eager），
   现在 graph 失败直接 `raise`。MoE 那边早有这条规矩（eager/serve 整段删除），
   这边一直双标。

2. **`auto_threads` 从 64 降到 16。** 单块 NVMe 在 QD=8..16 饱和；64 个线程只是
   抢 `_insert_locked` 那一把锁。实测预热期间 64 核全 100%、GPU 0%。
   `TI_READ_THREADS` 仍可覆盖。

## 与理论值的关系

| 口径 | 每步 | 吞吐 |
| --- | --- | --- |
| 显存带宽下限 | 5.2 ms | 92 tok/s |
| 已知 kernel 效率（8GB 卡实测外推） | 38 ms | 26 tok/s |
| **实测 graph** | **21.2 ms** | **47.2 tok/s** |

实测比「已知 kernel 效率」还快，说明那个 0.95ms/层 的外推是保守的 ——
L20 的 864 GB/s 比 8GB 笔记本卡快得多，MoE 单层实际不到 0.53ms。

离带宽下限还有 4 倍。下一步是做四项分解（MoE / attn / hc / 同步），
看这 21ms 里 MoE 占多少。
