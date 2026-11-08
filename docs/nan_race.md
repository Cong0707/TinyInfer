# NaN 的性质：竞态，不是数值溢出（2026-11-08 12:29）

## 决定性对比

```
无拦截：   PLAIN n=5 finite=False  nan=129280   （整个最后一行全 NaN）
有拦截：   L4 out finite=True  absmax=4.688      （完全正常）
```

拦截函数里做的唯一一件「额外」的事是 `torch.cuda.synchronize()`。
**加了同步就正常，不加就 NaN —— 这是竞态，不是数值问题。**

## 排除的数值猜想

之前几轮怀疑过的全部可以收回了：

- 「L2 输出 940 是数值爆炸」—— n=3 也有 940，logits 仍正常（17.794）
- 「rms_norm(eps=1e-20) 放大误差」—— attention 内部 `x_absmax` 只有 0.12-0.15
- 「重尾分布」—— 那是 `h`（block 级）的分布，attention 实际看到的是 rms_norm 后的值
- 「MoE routed 放大」—— 拦截后 L4 的 attention 输出 4.688，正常

## 竞态的三个候选

1. **`_gpu_y_buf` 复用**（`moe.py:655`）：分批路径里每批重建，
   `_out.clone()` 在 `torch.cuda.synchronize()` 之后，理论上安全；
   但 `moe.forward` 返回的 `_o` 是 view，**如果调用方在下一批 DMA 覆盖前还没读完**就悬空。
2. **槽位表与 DMA**：`commit()` 写的槽位表是主线程写的，kernel 异步读。
   本机只有 11 个槽位、每层 44 行 x 6 = 264 个专家请求，**必然反复覆盖同一个槽位**。
   覆盖时如果前一个 kernel 还没读完，读到的是新专家的字节 → 数值乱。
3. **`_sids_buf` / `_launch_keep` 复用**（`moe.py:552-554` 注释已经警告过）：
   driver-API 的 kernel 不认 Python 引用，跨批共享状态全靠 synchronize。

候选 2 最可疑：**这正是「槽位覆盖即回收」策略的固有风险** ——
用户要的覆盖式分配（不专门释放、直接覆盖）在单卡 14 个槽位时，
每步 264 个请求 / 14 个槽位 = 每个槽位被覆盖约 19 次。

## 验证方法（下一轮）

```python
# 在 moe 的 slot commit 之后、kernel launch 之前插 synchronize
# 如果 NaN 消失，就确认是槽位覆盖竞态
```

或者在 `_claim_slot_locked` 覆盖槽位前，对该槽位上一次的使用插一个 event wait。

## 为什么这很重要

如果确认是槽位覆盖竞态，那么**8GB 卡上的 decode 也可能是错的**（只是 4 层时
槽位压力小没触发）。这直接关系到「引擎能不能在低预算机器上撑并发」——
并发越多，槽位压力越大，这个竞态越容易命中。
