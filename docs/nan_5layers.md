# 8GB 卡的可运行层数：n=4 正常，n>=5 输出 NaN（未解决）

## 现象（已用「一进程一引擎」的纪律复测，排除测量污染）

```
RESULT n= 4 finite=True  absmax=  17.704 free= 1973MB chunk=8
RESULT n= 5 finite=False absmax=     nan free= 1298MB chunk=8
RESULT n= 6 finite=False absmax=     nan free= 1007MB chunk=8
RESULT n= 8 finite=False absmax=     nan free=  547MB chunk=8
```

**不是显存不足**：n=5 时还有 1298MB 空闲。是计算产生了 NaN。

## 已经定位到的

`_forward_chunk` 返回的 `_last_hidden` 已经是 NaN：

```
last_hidden finite=False absmax=nan shape=(1, 11, 4, 5120)
```

但**逐层直接调 `block.forward` 时每一层都有限**：

```
input   finite=True  absmax=0.453
layer 0 finite=True  absmax=1.273
layer 1 finite=True  absmax=230.000     <- 跳了 180 倍
layer 2 finite=True  absmax=208.000
layer 3 finite=True  absmax=202.000
layer 4 finite=True  absmax=188.000
```

所以问题不在「某一层算坏了」，而在 `_forward_chunk` 与「逐层手调」之间的**差异**：

1. `_forward_chunk` 里 `pre_mix` 是每步重新构造的（`zeros` + `[:, :, 0] = 1.0`），
   手调时我也这么做 —— 这一项一致。
2. **engram**：`engram_layer_ids = [1, 14]`。第 1 层是 engram 层，而它的 absmax
   从 1.27 跳到 230，是明显的异常点。`_forward_chunk` 会走 engram 的完整路径
   （包括 host 侧的投影），手调 `block.forward` 时未必走同一条。
3. `_forward_chunk` 有 `hashes` / `hash_ids` 的额外输入。

## 配置里顺带确认的事实

```json
"rms_norm_eps": 1e-20,
"compress_ratios": [0, 0, 2, 2, 2, ...]      // 不是 0x5，只有前两层是 0
"engram_layer_ids": [1, 14],
"kv_source_layer_ids": [2, 8, 14, 20],
"index_source_layer_ids": [2, 8, 14, 20, 24, 28, 32, 36],
"sliding_window": 128,
"tie_word_embeddings": false,
"rope_scaling": {"rope_type": "yarn", "factor": 16, "original_max_position_embeddings": 65536}
```

注意 `rms_norm_eps = 1e-20` 极小：任何接近 0 的方差都会放大成 inf/NaN。
这是高优先级嫌疑 —— 第 1 层 absmax 230 就是「范数被压得极小」的典型表现。

## 下一步

1. 对比 `_forward_chunk` 与手调在**第 1 层（engram 层）**的输入输出差异。
2. 检查 `engram` 的投影在序列长度为 11 时是否走了未覆盖的分支。
3. 单独测「ayers=[0,1]」（不含第 2 层）看是否已经 NaN —— 那就能确定是 engram。

**在那之前：8GB 卡上能可靠跑的是 4 层，5 层起输出 NaN。**
