# 最小复现：prefill -> reset 会让引擎进入永久坏状态（2026-08-08 13:41）

## 决定性实验

同一个引擎，四种调用序列（每次开头都 reset）：

```
prefill+reset   finite=False
prefill only    finite=False
reset only      finite=False     <- 注意：这一轮之前已经跑过 prefill+reset
neither         finite=True      <- 这是整个进程的第一次调用
```

看起来「reset only 也坏」，但**换一个干净进程单独跑 reset→1token 全部正常**：

```
after reset: N= 1 finite=True   absmax=17.00
after reset: N= 2 finite=True   absmax=16.30
after reset: N=11 finite=True   absmax=20.40
```

## 真正的结论

**`prefill` 之后再 `reset`，引擎进入永久坏状态**：

```
prefill(11 tok) -> reset() -> 1 tok      => NaN
prefill(11 tok) -> reset() -> 2/4/11 tok => 也 NaN（同一进程后续全坏）
干净引擎 -> reset() -> 任意长度          => 正常
```

一旦 `prefill + reset` 发生过，**这个引擎实例之后的所有 forward 都是 NaN**，
即使再 reset 也救不回来。

## 为什么这很重要

这是**引擎无法连续服务多个请求**的直接证据：

```
请求1: reset -> prefill -> decode   （正常）
请求2: reset -> prefill -> decode   （请求1 结束时如果 reset 过，请求2 全坏）
```

而 sched.py / server.py 的正常流程正是「每个请求 reset 一次」。

## 这解释了 test_engine_usable 的失败

`test_three_turns_are_all_finite` 通过（3 轮 reset→prefill→argmax 都是 finite），
`test_output_is_not_degenerate` 失败（在它之后跑，引擎已被污染）。

## 下一步

找出 reset 破坏了什么。已知 reset 会做：

- 清 `_last_routing`、`_flat_ev`、`_flat_done`、`_flat_slot_owner`
- `hashes.reset()`
- 每层 `b.attn.reset()` / `compressor.reset()` / `indexer.reset()`
- 清 `_bf16_cache`、`_expert_cache`
- 清 `shared_attn` 的若干属性
- streamed 层的 `b.to("cpu")` 再重建

最可疑的是最后一条：**streamed 层被搬到 CPU 再重建**。如果重建不完整，
或者常驻层和 streamed 层的界限变了，就会读坏权重。

验证：在 `prefill+reset` 之后检查 `self._stream_layers()` 是否和 reset 前一致。
