# 分批 NaN：本轮又排除了三个假设（2026-11-08 12:57）

## 本轮试过并排除的

| 尝试 | 结果 |
| --- | --- |
| 每批独立分配 `_gpu_y_buf`（结构性隔离）| NaN 不变，**已回退** |
| 把 `synchronize()` 移到 `clone()` 之前 | NaN 不变（保留，顺序本身对）|
| 加任何 hook 读取 buffer 形状 | NaN 消失（掩盖效应）|

## 代码里发现的一个真实隐患（但未必是本次 NaN 的成因）

`_sids_buf` 被三处共享：

```python
618  sids = _sb[:_sn]                       # 同一块 buffer
619  torch.index_select(tab, ..., out=sids) # 用途 1：给 _miss_scan
638  _nmiss = self._miss_scan(sids, ...)    # GPU kernel 异步读 sids
646  torch.index_select(tab, ..., out=sids) # 用途 2：给最终 kernel，覆盖同一块
713  fd(... sids ...)                       # 最终 kernel 读 sids
```

第 646 行**覆盖**第 619-638 行刚被 `_miss_scan` 读过的 buffer。如果 `_miss_scan`
的 kernel 还没跑完，这次覆盖会与它重叠。

加 hook 时，打印引入了隐式同步，掩盖了这个重叠 —— 这解释了为什么
「加打印就正常」。

**但**：改成每批独立 `_gpu_y_buf` 后 NaN 仍在，说明 `_gpu_y_buf` 不是（唯一的）
点；`_sids_buf` 的重叠更可能是真正的成因。

## 下一步（明确的修法）

给 `_sids_buf` 也做双缓冲：`_miss_scan` 用一个，最终 kernel 用另一个，
两者不重叠。改动很小：

```python
_sids_scan = ...    # 给 miss_scan
_sids_main = ...    # 给最终 kernel
```

但更简单的做法：**在 646 行重算 sids 之前，等 `_miss_scan` 的 kernel 完成**。
`_miss_scan` 内部本来就要把 miss 计数拉回主机（第 640 行 `.tolist()`），
那已经是一次同步点 —— 所以理论上不重叠。

需要确认 `_miss_scan` 的 `.tolist()` 是否真的同步了整个 stream。
