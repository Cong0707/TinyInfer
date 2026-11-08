# reset() 之后 decode 出 NaN：定位到 MoE 单行路径（2026-11-08 13:25）

## 精确到组件

```
E0 attn_in    finite=True  absmax=0.3008
L0 attn_out   finite=True  absmax=2.5625
L0 hcpost1    finite=True  absmax=0.5312
L0 ffn_in     finite=True  absmax=0.5586     <- MoE 输入正常
L0 ffn_out    finite=False absmax=nan        <- MoE 输出 NaN
```

**reset 之后，MoE 在 1 行输入上输出 NaN。** attention 和 hc_post 都正常。

## 触发条件

```
prefill(17 行) -> reset() -> decode(1 行)   =>  NaN
prefill(17 行) -> decode(1 行)              =>  正常
干净引擎 -> decode(1 行)                    =>  正常
```

所以需要 **prefill + reset + 单行** 三个条件同时成立。

## 已试过但无效的修复

| 尝试 | 结果 |
| --- | --- |
| `reset()` 里对 tensor 用 `zero_()` 而不是置 None | NaN 不变（**已保留**，语义更正确）|
| `_ascx.zero_()` 清零复用 buffer | NaN 不变（**已回退**）|

## buffer 实测

```
prefill: rows=17 n_exp=102  aq_buf=None     asc_buf=None
decode:  rows=1  n_exp=6    aq_buf=522240   asc_buf=16320   <- 复用 prefill 的大 buffer
```

`_aq_buf` 和 `_asc_buf` **尺寸只增不减**（`numel() < n_exp*_nk` 才重建），
所以 decode 用的是 prefill 留下的那一块。写入只覆盖前 `_rows*_tk*_nk` 段，
但清零没能修好 —— 说明 kernel 读的不是这块，或者还有别的地方。

## 下一个探针（明确）

在 `_forward_gpu_only` 里，对 1 行输入打印**最终 launch 的全部参数**：

- `scr` 切片长度（`n_exp*(nmid + nmid>>5)`）
- `aq` / `asc` 的实际 numel（不是 buffer 的）
- `sids` 的 6 个值（应该是 6 个有效槽位号，不是 -1 也不是 2^31）
- `blob`、`off[4]`、`off[5]`

重点看 **`sids` 里有几个 -1**：如果专家没搬进来（reset 清了 `_expert_cache`，
而 `_miss_scan` 认为「已就绪」），kernel 会读到 `slot_id = -1`，地址变成负偏移。

## 为什么这个 bug 重要

真实服务里 `reset()` 在**每次新会话**时调用。所以「新会话的第一次 decode」
会命中它 —— 这不是边缘情况，是主路径。
