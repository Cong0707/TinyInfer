# reset 后 decode NaN：同步位置全部无效（2026-11-08 13:29）

## 三个位置的同步都不管用

```
WHERE=none     decode finite=False
WHERE=before   decode finite=False      <- MoE 调用前同步
WHERE=after    decode finite=False      <- MoE 调用后同步
```

**不是异步/生命周期问题。** 加同步不改变结果，说明 NaN 是**确定性地算出来的**。

## 已经排除得差不多的清单

| 项 | 证据 |
| --- | --- |
| attention | 所有调用 out_finite=True，索引在界内 |
| hc_post / hc_mixes | 逐段 finite |
| sids 有 -1 | prefill 有 55 个 -1 却正常 |
| `_aq_buf` / `_asc_buf` 尺寸 | 精确匹配，清零无效 |
| `_gpu_y_buf` 跨批复用 | 每批独立分配无效（已回退）|
| 异步 / 生命周期 | 三个位置同步全部无效 |
| 分批 | 已改成默认不分批，NaN 仍在（在 reset 场景下）|

## 唯一的确定事实

```
L0 ffn_in   finite=True   absmax=0.5586
L0 ffn_out  finite=False  absmax=nan
```

MoE 拿到有限输入，确定性地算出 NaN。

## 剩下的可能

1. **专家权重本身是 NaN** —— reset 清了 `_expert_cache`，重建时从槽位取字节；
   如果槽位里是 NaN（比如前一次 prefill 写坏了），MoE 就是拿坏权重算。
   但 prefill 之后直接 decode 是正常的，说明槽位在那时是好的……
   除非 **reset 之后槽位被 evict/reload 了**。
2. **`_expert_cache.clear()` 之后，`_make_expert` 重建的对象指向的槽位不对**。

## 最直接的验证

在 reset 之后、decode 之前，直接检查第一个专家的权重是否 finite：

```python
rec = moe._expert_store.fetch(0, [eid], cache=True)[0]
ex = moe._make_expert(rec)
print("weight finite:", torch.isfinite(ex.w1.weight.float()).all())
```

如果权重含 NaN，root cause 就在槽位/加载路径，不在 MoE 计算。
