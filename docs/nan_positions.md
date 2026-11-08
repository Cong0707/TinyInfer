# NaN 只在位置 0 和 10（不是全序列）2026-11-08 12:34

## 新发现（推翻了「整条链数值失控」的判断）

```
positions with NaN: [0, 10]
nan 元素数: 129280 = 正好一个 vocab 行
```

5 层模型的 11 个 token 里，**只有第 0 位和第 10 位的 logits 是 NaN**，
中间 1-9 全部正常。

这否定了之前所有「数值放大」的解释：如果真是 rms_norm/MoE 数值爆炸，
应该是全序列一起坏，而不是精确地坏在首尾两个位置。

## 已排除

| 项 | 结果 |
| --- | --- |
| `window_idxs_ring(0, 11, w=128)` | 索引正常，pos0 有 1 个有效槽，pos10 有 11 个 |
| 压缩组号 | pos0 -> group0，pos10 -> group5，都合法 |
| `hc_mixes` (attn_pre/post/comb) | 位置 0/5/10 全部 finite，absmax 0.1-0.97 |
| 逐层同步 | sync_after = -1/2/3/4 全都仍然 NaN，**不是层间竞态** |
| 槽位 idle 检查（我加的那版） | 无效，已回退 |
| L0-L3 out | 全部 finite |
| L4 out | NaN |

## 唯一的有效干预

拦截 `MLACompressedAttention.forward` 并在其中 `torch.cuda.synchronize()`，
结果完全正常（L4 out absmax=4.688）。
但**在 Block.forward 之后同步无效** —— 所以竞态（如果存在）在 attention **内部**，
不在层之间。

## 首尾位置的共同点

11 个 token，位置 0 和 10：

- 位置 0：`window` 里只有 1 个有效槽位（它自己）
- 位置 10：最后一个位置，压缩组写到 group5（`compress_len = 11//2 = 5`，
  而 group5 的写入可能刚好越界）

`compress_len = end_pos // ratio = 11 // 2 = 5`，合法的组号是 0..4。
**位置 10 属于 group5，但 compress_len 只有 5 —— 它读不到自己所在的组**。
这不会产生 NaN，但会读到未初始化的组。

## 下一个探针（明确）

1. 在 L4 的 attention 内部，分别对位置 0 和位置 10 打印：
   `win_idxs[0, pos]`、`compress_len`、`compress_idxs[0, pos]`。
2. 检查 `compress_kv[:, :compress_len]` 里有没有 NaN/Inf（不是全体，是按组逐查）。
3. 用 12 个 token（compress_len=6，位置 11 属于 group5 且 compress_len>5）
   再跑一次，如果 NaN 位置变了就确认是组边界问题。
