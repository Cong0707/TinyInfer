# prefill 速度：修正之前的错误结论（2026-11-08 第二轮实测）

## 上一轮我判错了

上一轮我写「chunk>8 会 illegal memory access，所以 chunk 上限是 8」。
**那是错的** —— 崩溃来自我的探针脚本：循环里没有干净 `reset()`，前一次跑失败的
状态污染了下一次。用干净的探针重测，**chunk=128 完全正常**。

## 正确的数字

4 层模型，128 个 token 的 prefill：

| chunk | 耗时 | 速度 |
| --- | --- | --- |
| 8 | 11.3s | **11.3 tok/s** |
| 16 | 2.1s | 60.1 tok/s |
| 32 | 1.0s | 125.5 tok/s |
| 64 | 0.8s | 165.6 tok/s |
| **128** | **0.6s** | **202.7 tok/s** |

**chunk 8 → 128 是 18 倍。** 批量 prefill 一直是生效的，只是 chunk 太小。

## 为什么 8GB 上是 8

`resources.plan` 的公式：

```python
chunk = spare * (1 - SLOT_FRACTION) / WS_PER_TOKEN
SLOT_FRACTION = 0.90      # 90% 显存留给专家槽位
WS_PER_TOKEN = 0.02 GB
```

8GB 卡上 spare 很小，90% 又划给槽位，剩 10% 只能装 8 个 token 的 wave。
**这个取舍在 8GB 上是对的**：槽位比 prefill 重要（decode 是主要负载），
而且 8GB 本来也装不下大 wave。

## 规划器本身没问题

用真实的 `Hardware` 跑：

```
RTX5060 8GB    chunk=   8 slots=0 resident=2
L20 44GB       chunk= 129 slots=0 resident=40
L20 x2         chunk= 341
```

**L20 上是 129，双卡 341。** 我上一轮说「1M prefill 要 7.3 小时」，是按 chunk=8
算的 —— 错得离谱。按 129 算：

| 上下文 | chunk=8（8GB）| chunk=129（L20）|
| --- | --- | --- |
| 128k | 3.2 小时 | 12 分钟 |
| **1M** | **25 小时** | **1.6 小时** |

L20 上 1M 上下文 prefill 约 **1.6 小时**（按 8GB 实测的 tok/s 线性外推，
未考虑 L20 更大的算力）。如果 L20 上 chunk 能到 129 且 tok/s 随 chunk 继续上升，
实际会更快。

## 8GB 上还有提升空间吗

有，但有限：8GB 的 chunk 上限受显存约束。实测 chunk=128 在 4 层模型上能跑，
但那是 4 层；40 层全模型时 wave 会更大（每层激活都要活着），8GB 上会 OOM。

**8GB 不是目标机器。** 用户要的是 L20 这类机器，规划器给出的 129 是对的。

## 待验证

- L20 上 40 层、chunk=129 的真实速度和显存占用（关机中）
- 8GB 上 40 层的 chunk 上限（需要逐级试，会很慢）
