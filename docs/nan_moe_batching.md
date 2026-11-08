# 根因确认：MoE 分批的跨批共享 buffer 污染（2026-11-08 12:50）

## 决定性实验

把 MoE 分批阈值改成 `TI_MOE_BATCH`，同一输入（11 行）跑不同批次大小：

| 批次 | 切分 | NaN 位置 |
| --- | --- | --- |
| 2 | 2x5+1 | **无** |
| 3 | 3+3+3+2 | [10] |
| 4 | 4+4+3 | [8, 9] |
| 6 | 6+5 | [10] |
| 8 | 8+3 | **无** |
| 11 | 11 | **无** |

**NaN 的位置随批次大小移动，而且 2/8/11 完全干净。**

这证明：问题不在数值、不在模型、不在某一层 —— 而在 **MoE 分批之间共享的 buffer**。

## 为什么 2 / 8 / 11 干净

看每批的行数：

```
BS=2:  2,2,2,2,2,1     <- 只有最后一批 1 行
BS=3:  3,3,3,2         <- 最后一批 2 行
BS=4:  4,4,3           <- 最后一批 3 行
BS=6:  6,5             <- 最后一批 5 行
BS=8:  8,3             <- 最后一批 3 行
BS=11: 11              <- 只有一批
```

BS=8 的最后一批也是 3 行，却是干净的；所以不是「最后一批行数」那么简单。

更可能的是**批次数量**：BS=2 有 6 批但干净，BS=3 有 4 批有问题 —— 也不成立。

暂时没有简单规律。**但事实足够明确：跨批共享状态是根因。**

## 涉及的共享状态（`moe.py` 分批循环）

```python
for _s0 in range(0, xf.size(0), _moe_bs):
    _o = self._forward_gpu_only(_x, xf[_s0:_s1], indices[_s0:_s1], weights[_s0:_s1])
    _outs.append(_o.clone())
    torch.cuda.synchronize()      # 注释说这是「唯一屏障」
```

每次 `_forward_gpu_only` 内部会重建/复用：

- `_gpu_y_buf`（按 `xf.shape`，形状随批变化就重建）
- `_sids_buf`（槽位编号）
- `_aq_buf`（激活量化）
- `_launch_keep`（保持 kernel 参数的引用）

注释里已经警告过：**「下一批换形状会重建并释放那块 buffer，前面几批的 view 就全悬空了」**，
而 `_o.clone()` 就是为了防这个。但 `.clone()` 在 `torch.cuda.synchronize()` **之前**执行 ——
如果 kernel 还在写 `_gpu_y_buf`，clone 拿到的就是半成品。

## 最直接的修法（下一轮）

把 `torch.cuda.synchronize()` 移到 `_o.clone()` **之前**：

```python
_o = self._forward_gpu_only(...)
torch.cuda.synchronize()          # 先等 kernel 写完 _gpu_y_buf
_outs.append(_o.clone())          # 再克隆
```

当前顺序是 clone 后同步，正好反了。
