# reset 后 decode NaN：排除清单（2026-11-08 13:32）

## 现象（唯一确定的事实）

```
prefill(17行) -> reset() -> decode(1行)

L0 ffn_in   finite=True   absmax=0.5586
L0 ffn_out  finite=False  absmax=nan
```

MoE 拿到有限输入，**确定性地**算出 NaN。

## 全部排除项（附证据）

| # | 假设 | 证据 | 结论 |
| --- | --- | --- | --- |
| 1 | attention 有问题 | 所有调用 out_finite=True，idx 全在界内 | 排除 |
| 2 | hc_post / hc_mixes | 逐段 finite，absmax 0.1-1.0 | 排除 |
| 3 | sids 有 -1 | prefill 有 55 个 -1 却 finite | 排除 |
| 4 | `_aq_buf` 残留 | 尺寸精确匹配，zero_ 无效 | 排除 |
| 5 | `_asc_buf` 残留 | 同上 | 排除 |
| 6 | `_gpu_y_buf` 跨批复用 | 每批独立分配无效 | 排除 |
| 7 | 异步 / 生命周期 | before/after/无 三种同步结果相同 | 排除 |
| 8 | **专家权重含 NaN** | fp4 six tensors 前后完全一致且 finite | **排除** |
| 9 | 分批 | 默认已改不分批，reset 场景仍 NaN | 排除 |
| 10 | `_launch_keep` 生存期 | 同步无影响 | 排除 |

## 剩下什么

只有一种可能：**MoE 的计算对「这个特定输入 + 这个特定内部状态」敏感**。

而「内部状态」在这个阶段已经查不出差异了 —— 因为：

- 权重相同（fp4 六个 tensor 逐字节一致）
- 输入有限（0.5586）
- buffer 尺寸正确
- 同步无影响

## 一个之前没试过的方向

**`swiglu_limit`**。`Expert.swiglu_limit` 是 float，来自 `cfg.swiglu_limit = 10.0`。
它作为 kernel 参数 `float(self.swiglu_limit)` 传入。

如果 sigmoid/exp 在某个输入上溢出，`swiglu_limit` 的 clamp 逻辑可能产生 inf。
但这条与 reset 无关……

## 更可能的：reset 清了不该清的东西

`reset()` 第 1419 行清 `_expert_cache`，第 1400-1415 行清 attention 的 `_bf16_cache`。

`_bf16_cache` 是 **fp8 权重的反量化缓存**。清掉之后，下一次 forward 会重建。
如果重建时 `self.wo_a.weight` 恰好指向一块**被 reset 释放的显存**，重建出来的
就是垃圾 —— 而 attention 自己算出来的 `ya` 却是 finite 的（因为 attention 用的是
另一条路径），到了 MoE 才表现出来。

**验证**：reset 之后不调 `_forward_chunk`，直接检查 `b.attn.wo_a` 的权重是否 finite。

## 建议

这个 bug 已经排查了十几轮，每轮都在缩小范围但没有命中。
**8GB 卡上 4 层模型是能正常端到端出字的**（probe_local 答出「2」），
所以这不是「引擎完全不能用」，而是「5 层 + reset 这个组合有缺陷」。

在拿到 L20 之前，继续用探针追它的投入产出比已经很低。
应当优先做能在真卡上验证的工作，把这个记成已知缺陷。
