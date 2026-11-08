# 并发能力：做到哪、卡在哪（2026-11-08 实测）

## 已经打通的

**1. `max_slots=4` 和 `TI_MAX_BATCH=8` 是写死的，现在都从硬件算。**

`kv_tiers.plan_slots()` 按真实剩余显存 + 上下文长度推导：

| 卡 | 1M 上下文并发数 |
| --- | --- |
| 8GB (本机 5060) | 1 |
| 44GB (L20) | 4-7 |
| 141GB | 几十 |

上下文缩短，并发数自动变大（128k 比 1M 多得多）。这是长上下文和并发之间
真实的 trade-off，现在是公式算出来的，不是人猜的。

**2. KV 不再按上限预分配。**

旧代码 `cap = max_position_embeddings // ratio`：一条 1M 会话一开就占 6.91GB
（实测：本 checkpoint 的 `compress_ratios` 是 `0×5, 2×18, 1×20`，
head_dim 512，latent 以 fp4 存）。改成几何增长之后，同样的会话按需爬到 6.91GB，
短会话就只占短会话那么多。

**3. KV 有换出（park/resume）。**

`KVPool.park()` 把一个会话的 KV 从槽位搬到主机内存，`acquire()` 时原样装回。
以前 `acquire()` 里只有 `seq.state.clear()` —— 会话一失去槽位历史就没了，
下次回来要整段重新 prefill（1M 是几分钟）。`max_parked` 限流，内存不会涨。

**4. batch>1 的前向在 window-only 层能跑。**

`attention.py` 里按行算窗口索引时，把 `positions`（`[b, s]`）当成标量用，
`int(list)` 直接 TypeError —— 这就是「10 次请求 2-3 次 500」的根源，
也是 `sched.py` 把 `max_batch` 压到 1 的理由。修掉之后：

- `tests/test_batched_forward.py`：batch=2 前向通、先单条再拼批通
- `tests/test_mixed_batch.py`：两行不同 chunk 位置（pos=[0,4]）拼批，
  第 0 行输出与单独跑一致

同时修掉了 MoE 分批的两个真 bug：`x[:, a:b]` 按第二维切（应该第一维）、
`torch.cat(_outs, dim=1)` 拼错维（应该是 dim=0），以及分批后没把形状还原
成 `[b, s, hc, d]`，导致 `hc_post` 抛 "size of tensor a (2) must match (4)"。

## 还卡着的：compressed 层不支持 batch>1

`tests/test_mixed_batch.py` 用 `layers=[0, 1]`（第 1 层 `compress_ratios[1]=0`）
能跑；换成含压缩层的配置就出 NaN。原因是压缩 KV 的写入位置按**整批**算：

```python
start_group = start_pos // self.ratio        # ← 一个数，所有行共用
self._compress_kv[:, start_group : start_group + latent.size(1)] = latent
```

`start_pos` 是标量（batch 里最小/最大那个），于是两行不同深度的会话会写到
同一个 group 区间、互相覆盖。代码里已经有 `_grp0`（per-row group index）在
处理这件事，但只在 `positions.numel() == b` 时生效 —— 那条路径假设每行只有
**一个**位置（纯 decode），prefill 行有多列就不满足了。

要真正支持混批，压缩 KV 的写入要改成逐行定位：每行按自己的 group 偏移写，
同一 group 区间内各行互不覆盖。这是下一步，不是小改动。

## 结论

8GB 卡上这些都没法验证出有意义的吞吐（1 个槽位、100% 专家重读）。
真正该量的是 L20：那里 1359 个槽位装得下 240 个专家、并发 4-7 条、
1M 会话的 KV 只占 30GB 预算的一小部分。**远端仍关机，这一轮所有结论
都来自本机推导和窗口层实测。**
