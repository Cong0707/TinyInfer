# 拼批与逐 token 的差异定位到 MoE 的 n_exp 依赖（2026-08-08 14:20）

## 逐层数据（N=8，比较第 7 个 token）

```
L0 attn:  batch 4.5000  token 4.5000  maxdiff=0.003906   <- 纯浮点噪声
L0 moe:   batch 2.5312  token 2.6250  maxdiff=0.109375   <- 4.3%
L0 整体:  batch 1.188   token 1.320   maxdiff=0.156250   <- 13%
L1:       maxdiff=0.112
L2:       maxdiff=0.190
L3:       maxdiff=0.218
```

**attention 一致，MoE 不一致。** 而且误差每层累积。

## 路由完全一致

```
batch row7 idx=[52, 177, 200, 251, 286, 338]
single     idx=[52, 177, 200, 251, 286, 338]   idx same: True
w maxdiff: 0.00000000
```

**专家选择相同、权重相同。** 差异只能来自专家计算本身。

## 直接对比同一行

```
batch(n_exp=48) absmax=0.0437
single(n_exp=6) absmax=0.0508
maxdiff=0.008545   相对 20%
```

绝对量级很小（0.04-0.05），但相对误差 20% —— 远超 bf16 的 ~0.4%。

## 这指向什么

`moe_fused` kernel 里 `n_exp` 参与：

- `for (int e = 0; e < n_exp; ++e)` —— 专家循环
- `sids[e]` / `weights[e]` 的索引
- `fd` 的 grid y 维（已修，按 `_rows` 缓存）

而 `n_exp = _rows * topk`。同一行在不同 `_rows` 下的 `e0 = row_d * topk` 不同，
但那只是**索引到不同的专家槽位**，值应该一样。

**除非**：`y` 的累加顺序变了，或者某个专家的 `slot_ids` 在两批里指向不同槽位，
而那两个槽位的字节不完全相同（槽位复用导致的旧数据残留）。

## 最可疑的：槽位复用

8 行需要 48 个槽位，但池子只有 14 个 —— **100% miss**，专家被反复搬入搬出。
同一行在两次运行中，专家可能落在不同槽位，而**槽位里的字节可能不是刚搬的那个**。

这是「覆盖式槽位」在高压力下的真实代价。

## 验证

对比两次运行里同一专家的槽位字节：

```python
slot_b = sl._by_key[(layer, expert)]
slot_s = sl._by_key[(layer, expert)]
buf_a = sl.buffer(slot_b)
buf_b = sl.buffer(slot_s)
print(torch.equal(buf_a, buf_b))
```

如果不等，就确认了槽位里是脏数据。
