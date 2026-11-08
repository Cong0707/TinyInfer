# 关键转折：940 是正常的，问题在 L4 自身（2026-11-08 12:27）

## n=3 和 n=5 的对照

```
n=3:
  L0 in 0.453 -> out 1.273
  L1 in 1.211 -> out 1.047
  L2 in 1.047 -> out 940.000
  logits  finite=True  absmax=17.794      <- 940 之后仍正常

n=5:
  L0 in 0.453 -> out 1.273
  L1 in 1.211 -> out 1.047
  L2 in 1.047 -> out 940.000              <- 与 n=3 完全一致
  L3 in 940   -> out 808.000
  L4 in 808   -> out nan
```

## 这推翻了我上一轮的结论

上一轮我写「L2 把 1.047 放大到 940，是 NaN 的源头」。**错了。**
n=3 时 L2 同样输出 940，最终 logits 却是正常的 17.794。

940 是 `[1, 11, 4, 5120]` 里 hc 四分量中的极值，后续 `hc_post` + `rms_norm` 会把它
归约到正常量级。所以 **940 不是异常，是这条模型链路本来就有的量级**。

（这也解释了为什么 ffn_norm 之后是 0.828 —— rms_norm 把 940 压回 0.8，正常。）

## 真正的问题：L4 自身

```
L4 in = 808（有限）  ->  L4 out = NaN
```

L3 和 L4 是同一类层（`MLACompressedAttention`, ratio=2, is_kv_source=False），
**L3 正常、L4 坏**。两者唯一的结构差异是它们在 `index_source_layer_ids` 里的位置：

```json
"index_source_layer_ids": [2, 8, 14, 20, 24, 28, 32, 36]
"kv_source_layer_ids":    [2, 8, 14, 20]
"candidate_source_layer_id": 20
"engram_layer_ids": [1, 14]
```

L2 是 kv_source + index_source；L3 是纯 consumer；L4 也是纯 consumer。
但 L4 的 `rt`（shared state）由 L2 写入。**如果 L4 读的时候 compress_len 算错，
就会读到未初始化的组** —— 那是 `_compress_kv` 分配时没清零的内存。

## 下一个探针（明确）

1. 在 L4 的 attention 里打印 `compress_len`、`rt.compress_kv.shape`、
   `rt.compress_kv[:, :compress_len]` 的 finite 状态。
2. 对比 L3 的同一组值。
3. 如果 L4 的 `compress_len` 大于实际写入的组数，就确认是越界读。
