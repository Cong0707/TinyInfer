# reset() 之后 decode 出 NaN：状态被清，重建不完整（2026-11-08 13:20）

## 精确对照

```
MODE=none   decode finite=True   absmax=14.188    <- prefill 后直接 decode，正常
MODE=reset  decode finite=False  absmax=nan       <- 中间加了 reset，NaN
MODE=sync   decode finite=True   absmax=14.000    <- 只同步，正常
```

**prefill 后直接 decode 是正常的。是 `e.reset()` 把它弄坏的。**

这更正了我上一轮的判断 —— 我上一轮说「prefill 污染 decode」，实际是
**reset() 污染 decode**，而 probe_parity2 里恰好调了 reset。

## reset 之后 shared_attn 的状态

```
after reset#1  _mask_cache=ABSENT  compress_kv=None  topk_idxs=None  index_k=None  candidates=None
after prefill  _mask_cache=((0,17,8,2), bool tensor)  compress_kv=((1,1024,512), True)
after reset#2  _mask_cache=None  compress_kv=None  topk_idxs=None  index_k=None  candidates=None
```

**reset 把全部共享状态清成 None，decode 之后仍然是 None** ——
说明 1 行输入的 `_forward_chunk` **没有重建这些状态**。

## reset 里那段代码的问题

```python
for _attr in ("_mask_cache", "candidates_by_chunk", "candidates",
              "compress_kv", "topk_idxs"):
    if hasattr(_sa0, _attr):
        try:
            getattr(_sa0, _attr).clear()      # tensor 没有 clear()
        except AttributeError:
            setattr(_sa0, _attr, None)        # 于是整个 tensor 被丢弃
```

tensor 没有 `.clear()`，AttributeError 被吞后走 `setattr(..., None)`。
**这不是「清空」，是「销毁」**。

已改成：tensor 走 `zero_()`（保存储）、容器走 `.clear()`、其它才置 None。
但**改了之后 NaN 仍在** —— 说明问题不在「被销毁」，而在「重建」。

## 真正的问题

`_forward_chunk` 在 1 行输入时，如果没有可复用的 `compress_kv`，它应该：

1. 重新分配（或复用 `self._compress_kv`）
2. 或者走「没有压缩历史」的降级路径

实测 decode 之后 `compress_kv` **仍然是 None**，说明两条路都没走 ——
而是走了某个「读到 None 就当空 tensor 用」的路径，产生了 NaN。

## 下一个探针

在 `compressed_attention.forward` 里，当 `rt.compress_kv is None` 时打印分支，
看它走了哪条路。代码里有：

```python
elif self.rt.compress_kv is None:
    compress_kv = compress_idxs = None    # 退化成「本步不带压缩历史」
```

这条注释说「至少不炸整批」，但它对 **consumer 层**可能不成立 ——
consumer 层的压缩注意力如果没有压缩历史，sink 之外的 logit 可能全是 -inf。
