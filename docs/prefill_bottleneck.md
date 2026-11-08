# prefill 的真实速度：一个必须先修的瓶颈（2026-11-08 实测）

## 数字

```
batch prefill 64 tokens: 6.71s  = 9.53 tok/s
per-token  forward:      0.104s/token = 9.59 tok/s
speedup of batch over per-token: 1.0x        <- 完全没有并行
```

**送 64 个 token 和逐个送 64 次一样快。** 这不是「慢」，是「批量根本没生效」。

## 根因链

1. `prefill_chunk` 被硬件计划设成 **8**（`[tinyinfer] ... chunk=8`）。
2. `_forward_impl` 里 `step = chunk or max(self.prefill_chunk, 32)`，
   但硬件计划给的是 8，且 `max(..., 32)` 只在 `chunk` 为 None 时生效 ——
   实际走的是 8。
3. 于是 1M 上下文的 prefill = 131072 个 chunk，每 chunk 8 个 token。

## 为什么 chunk 上不去：真正的 bug

单层时 chunk=16 完全正常（logits 形状对，无报错）。
**4 层时 chunk>8 直接 illegal memory access。**

这说明问题在 **streamed 层**（层权重反复搬运那条路径）的 buffer 大小
和 chunk 绑定，而不是 attention 或 MoE 本身。

## 为什么这必须优先修

按当前速度：

| 上下文 | chunk=8 的 prefill | L20（估算快 4x）|
| --- | --- | --- |
| 1k | 1.7 分钟 | 26 秒 |
| 128k | 3.7 小时 | 56 分钟 |
| **1M** | **29 小时** | **7.3 小时** |

用户的要求是「真实的请求是 1m 的上下文」。**29 小时不可接受，7.3 小时也不可接受。**

对比：如果 chunk 能到 256（代码注释里说「256 是最快的测量点，3.5x chunk=8」），
1M 的 prefill 就是 131072/256 x (256/9.53) ≈ 3.6 小时（L20 上约 54 分钟）。
仍然慢，但至少是「能等」和「不能等」的区别。

## 下一步（明确）

1. 定位 streamed 层的 buffer 为什么和 chunk 绑定 —— 在 `_flat_setup` /
   `_stream_layer` 里找按 chunk 尺寸分配的 buffer。
2. 把它改成按 `prefill_chunk` 的实际上限分配（或按需增长），
   然后让 `resources.plan` 的 `prefill_chunk` 回到一个合理的值（128-256）。
3. 用 `probe_prefill.py` 验证：`chunk=32/64/128` 都必须不崩，且 tok/s 随 chunk
   上升。

**在那之前，1M 上下文是「能跑但等不起」的状态，不要声称支持。**
