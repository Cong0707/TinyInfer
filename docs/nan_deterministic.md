# NaN 是确定性的位置 bug，不是竞态（2026-11-08 12:41）

## 决定性验证：5 次独立进程，结果完全一致

```
TRIAL N=11 bad=[10]
TRIAL N=11 bad=[10]
TRIAL N=11 bad=[10]
TRIAL N=11 bad=[10]
TRIAL N=11 bad=[10]
```

**这不是竞态。** 我上一轮说「加 synchronize 就正常 → 竞态」是错的 ——
那个拦截改变的是内存分配/复用模式，让未初始化的数据碰巧是 0，
所以看起来「修好了」。真正的 bug 是确定性的。

## 完整长度扫描（每个 N 独立进程）

| N | compress_len | NaN 位置 |
| --- | --- | --- |
| 6 | 3 | 无 |
| 7 | 3 | 无 |
| 8 | 4 | [6, 7] |
| 9 | 4 | 无 |
| 10 | 5 | [9] |
| **11** | **5** | **[10]** |
| 12 | 6 | 无 |
| 13 | 6 | 无 |
| 14 | 7 | 无 |
| 16 | 8 | 无 |

规律：**坏的位置总是序列的最后一位或最后两位**，且可重现。

N=8 和 N=9 的 compress_len 都是 4，但一个坏一个好 —— 说明不是简单的
`compress_len` 边界问题，而是**序列末尾位置的窗口/压缩读取**有缺陷。

## 已排除（本轮新增）

| 项 | 证据 |
| --- | --- |
| `compress_kv` 未初始化 | 写入组数 = compress_len，`nan_elems=0`，全 finite |
| `window_idxs_ring` 越界 | N=8/11/12/13 的尾部索引都正常，max 等于 pos |
| 层间竞态 | 试了 sync_after=-1/2/3/4，全部仍 NaN |
| 槽位覆盖竞态 | 加了 idle 检查无效（已回退）|
| 5 次重复的随机性 | 完全一致，确定性 |

## 下一步（明确）

在 L4（或任一 consumer 层）的 attention 内部，针对 **最后一个位置** 打印：

```python
pos = N - 1
print("win_idxs[pos]:", win_idxs[0, pos].tolist())
print("compress_idxs[pos]:", compress_idxs[0, pos].tolist() if compress_idxs is not None else None)
print("compress_len:", compress_len)
print("kv_att finite at those slots:", ...)
```

重点看**最后一个位置读的 kv 槽位里有没有 NaN**：
如果 kv 本身有 NaN，说明是写入侧；如果 kv 干净但输出 NaN，说明是 softmax 分母为 0。

## 注意

`sink`（attention sink）在稀疏 attention 里是「额外的一个 logit，没有 value」。
当最后一个位置的有效槽位很少时（窗口内只有 1-2 个），softmax 的分母可能极小而溢出。
这是下一个要重点查的方向。
