# 并发能力：拼批到底支不支持（2026-11-08 修正）

## 结论先行

**attention 和 MoE 的拼批都是正确的。** 单行与 batch=2（两行同一份数据）的输出
逐值相同：

```
attention row0 vs single = 0.000000   （严格相等，测试断言 == 0.0）
MoE/ffn   row0 vs single < 1e-2      （fp32 累加顺序带来的浮点噪声）
```

`tests/test_batch_consistency.py` 锁住这两条。

## 之前几轮为什么得出了相反的结论

前几轮测出 attention 拼批差 4.70 / 2.90 / 0.52，结论是“一份 window buffer
装不下两个历史、架构限制”。**那个结论是错的，错在测试方法。**

真正有两个 bug 和一个调用约定：

### bug 1：window buffer 没清零（已修）

```python
buf = kv.new_empty(b, self.window, self.head_dim)   # 未初始化内存
```

环形 buffer 一个 token 只写一格，剩下 127 格是分配器的垃圾。窗口索引会扫到
这些槽位；单行时索引恰好把它们排除，batch>1 时每行有效槽位不同，垃圾就混进
结果。实测两行**完全相同**的输入，输出差 2.90，且 `buf[0]` 有 58 个非零槽位
（只写了 1 个 token）。改成 `torch.zeros` 后差值归零。

### bug 2：window 按同一个 start_pos 写（已修）

```python
buf[:, start_pos % self.window] = kv[:, 0]   # 所有行写同一格
```

改成按每行自己的位置写（`index_copy_` 逐行）。

### 调用约定：start_pos 必须等于本批最小位置

`positions` 是每行每个位置的绝对位置，`start_pos` 是本批的基准位置。
两者不一致时，单行走 `window_idxs_ring_cached(start_pos)`、batch 走
`window_idxs_per_row(positions)`，算出来的窗口索引不同 —— 看起来像“拼批不支持”，
其实是调用方传错了参数。所有失败的测量都源于这个（我传了 `start_pos=0`
而 `positions=[1]`）。

真实调用里 `engine.forward` 总是传 `start_pos=int(pos.min())`，所以不会不一致。

## 已修的三处 MoE 分批 bug

1. `x[:, _s0:_s1]` 按第二维切 → 应该按第一维（行）切
2. `torch.cat(_outs, dim=1)` → 应该 `dim=0`（分批后每批是 `[rows, dim]`）
3. 分批后没把形状还原成 `[b, s, hc, d]` → 上层 `hc_post` 抛
   "size of tensor a (2) must match (4)"

## 还没做的

- 压缩层（`compressed_attention.py`）的写入已改成逐行 scatter，但阅读侧
  `compress_kv = rt.compress_kv[:, :compress_len]` 仍按单一起点取；多行不同深度
  时是否完全正确**尚未验证**。
- 多卡 EP 真机验证（远端 L20 关机中）。
- KV 换出/换入接到 forward 路径（接口和策略已就位）。

## 关于「8GB 卡上能不能并发」

能，但没意义：`plan_slots()` 按真实显存算出来是 1。并发要靠 L20（4-7 条）
或者更大的卡。这一轮的价值是**证明拼批路径本身是对的**，去掉了一个
以为存在、实际不存在的障碍。
