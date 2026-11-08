# NaN 的确切边界：含 kv_source 层或 engram 层组合（2026-11-08 12:13）

## 用「一进程一引擎」的纪律测出的边界

```
layers=0            finite=True
layers=0,1          finite=True
layers=0,1,2        finite=True
layers=0,1,2,3      finite=True
layers=0,1,2,3,4    finite=False      <- 分界

layers=3,4          finite=True       <- 两条 consumer，正常
layers=2,3          finite=False      <- 含 kv_source (layer 2)
layers=2,4          finite=False
layers=2,3,4        finite=False
```

两个现象：

1. `[0..3]` 正常，`[0..4]` NaN —— **多一层就坏**
2. 只要含 layer 2（`kv_source`）就 NaN，但**必须同时有 layer 0/1**？不：
   `[2,3]` 直接 NaN，`[3,4]` 正常。

## 逐层手调全部正常

```
layer 0 finite=True absmax=1.273
layer 1 finite=True absmax=230.000
pre-attn  finite=True absmax=0.7773      (layer 2 入口)
attn out  finite=True absmax=8.312       (layer 2 出口)
  rt.compress_kv  finite=True absmax=1.6875
  rt.index_k      finite=True absmax=2.1406
  rt.topk_idxs    finite=True absmax=132.0
```

**手调 `block.forward` 逐层跑没有任何 NaN**，而 `engine.forward` → `_forward_chunk`
就 NaN。差异在 `_forward_chunk` 额外做的事：

## 差异点：engram

`engine._forward_chunk` 第 2360 行：

```python
h = b.engram.forward(h, hash_ids[b.engram.hash_index])
```

- `engram_layer_ids = [1, 14]`（config.json 确认）
- 我的逐层手调**没走这一行**
- `engram(h, None)` 直接抛 `NoneType has no attribute reshape` —— 它强依赖 hash_ids

所以：**手调与整段的差别就是 engram 层是否被执行**。

## 另一个可疑点：第 1 层 absmax 230

```
layer 0 -> 1.273
layer 1 -> 230.000      <- 181 倍跳变
layer 2 -> 208.000
```

正常 transformer 的残差流不会在一层内跳 181 倍。这一层正是 engram 层。
`rms_norm_eps = 1e-20` 意味着范数接近 0 时除法会炸成 inf，
而 bf16 的 inf 再参与后续运算就是 NaN。

## 还没做

1. 在 `_forward_chunk` 里对第 1 层前后打点，确认 NaN 首次出现在 engram 之后。
2. 检查 `hash_ids` 在只跑 4 层时是否为 None（`hashes` 需要完整层表？）。
3. 如果 engram 是根因：8GB 卡上要么关掉 engram（需确认不影响正确性），
   要么修 engram 的数值。

**当前可靠结论：8GB 卡上 4 层正常，5 层起 NaN，根因指向 engram 层。**
