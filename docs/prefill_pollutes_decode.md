# 关键：整批 prefill 之后，逐 token 路径失效（2026-11-08 13:15）

## 对照

**probe_tok.py**（干净引擎，只跑逐 token）：

```
pos  0..16 全部 finite  absmax 14-21
```

**probe_parity2.py**（先跑一次整批 prefill，再跑逐 token）：

```
tokens=17  batch_finite=True     <- 整批正常
  pos 0..8: token path NaN       <- 之后逐 token 全 NaN
```

## 这就是真实推理的路径

```
1. prefill: 一次送入整个 prompt（整批）
2. decode:  每次送入 1 个 token
```

**第 1 步污染了第 2 步。** 这是最该修的 bug —— 比多行 NaN 严重得多，
因为它是**每一次请求**都会走的路。

## 可能的机制

整批 prefill 之后，某些状态没被重置：

- 压缩 KV 的写入位置（`rt.compress_kv` / `compress_len`）
- 窗口环形缓冲的写指针
- `_gpu_y_buf` / `_sids_buf` 的形状（整批是 17 行，逐 token 是 1 行）
- 槽位表

`e.reset()` 被调用了吗？probe 里没调 —— **这可能是探针的问题**：
真实服务里 prefill 后直接 decode，不会 reset。

## 下一步

1. 先确认这是不是探针问题：在 prefill 和 decode 之间加 `e.reset()`，看是否恢复。
2. 如果不恢复，说明 reset 也清不掉，那是真的状态泄漏。
3. 如果 reset 后恢复，说明 prefill 留下的状态确实有害 —— 需要找出是哪一个。

## 注意

`e.reset()` 在真实服务里不能每步调用（它会清掉 KV）。所以即使 reset 能「修好」，
也说明存在真实缺陷。
