# reset NaN 的实际影响范围（2026-11-08 13:34）

## 真实使用模式下不会触发

多轮对话的实际序列：

```
turn0: reset() -> prefill(prompt) -> decode x N
turn1: reset() -> prefill(prompt) -> decode x N
turn2: ...
```

实测（4 层模型，同一引擎连续 3 轮）：

```
n=4 turn0 finite=True  argmax=82277
n=4 turn1 finite=True  argmax=105744
n=4 turn2 finite=True  argmax=105744
```

**三轮全部正常。** turn1 和 turn2 的 argmax 相同是合理的（两个问题的答案都是 4）。

## 触发 NaN 的序列是病态的

```
prefill(17 行) -> reset() -> decode(1 行)      => NaN
```

这只是我在探针里构造的序列 —— **先 prefill 一整个 prompt，不生成就 reset，
然后直接 decode 一个 token**。真实推理不会这么做：

- 要么 prefill 之后接着 decode（中间不 reset）→ 正常
- 要么 reset 之后重新 prefill 新 prompt → 正常

## 所以这不是「主路径 bug」

我上一轮说「reset 在每次新会话时调用，所以新会话第一次 decode 会命中它」——
**那个判断不准确**。新会话的顺序是 reset → prefill → decode，
reset 后面紧跟的是 prefill（多行），不是单行 decode。

## 仍然存在的真实问题

`reset()` 把 tensor 类型的共享状态置 None 是**语义错误的**（tensor 没有 `.clear()`），
已经修成 `zero_()`。这条改动是独立正确的，与 NaN 无关。

## 结论

| 场景 | 状态 |
| --- | --- |
| 4 层模型多轮对话 | **正常**（实测 3 轮）|
| 4 层模型端到端出字 | **正常**（probe_local 答「2」）|
| prefill → reset → 单行 decode | NaN（病态序列，真实推理不会走）|
| 5 层模型 | 8GB 显存不够（OOM，不是 NaN）|

**这个 NaN 是探针构造出来的病态序列触发的，不是生产路径的缺陷。**
继续追它的投入产出比很低；记录在案，等有真卡且能跑 compute-sanitizer 时再看。
