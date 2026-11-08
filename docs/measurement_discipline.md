# 测量纪律：引擎在一个进程里只能可靠地跑一次

## 为什么写这个

这一轮我在同一个进程里反复创建引擎、反复 forward，得出了**三个错误结论**：

1. 「chunk>8 会 illegal memory access」—— 实际是探针之间状态污染
2. 「n=8 时 free=1203MB 却输出 NaN」—— 实际是前一次运行的残留
3. 「first 4 个 token 单独跑 NaN」—— 实际是前一次 `full T=11` 的残留

每次都在文档里写了「这是真 bug」，然后下一条探针把它推翻。代价是几轮时间，
还有两份写错的分析。

## 规律

正确做法：**一个进程 = 一个引擎 = 一次测量**。

```python
# 对：每次运行一个独立进程
foreach ($n in 4, 8, 11) { python probe.py $n }

# 错：一个进程里循环
for n in (4, 8, 11):
    e = Engine(...)   # 第二个引擎开始就不可靠
    e.forward(...)
```

为什么：

- `Engine` 持有 CUDA context、pin 内存、流、graph 等进程级资源，析构**不可靠**
  （Python 的 `del` + `gc.collect()` + `empty_cache()` 实测回收不干净）
- 第二个引擎初始化时，第一个的残留会和规划器的假设打架（比如 free 显存读数）
- 同一个引擎上的第二次 `forward` 如果中间 `reset()` 不彻底，KV/槽位状态会串

## 已验证的对照

同一个探针，独立进程逐次运行：

```
first  1 tokens: finite=True absmax=17.00
first  4 tokens: finite=True absmax=20.25
first 11 tokens: finite=True absmax=19.95
```

全部正常。而一个进程里连跑，就出现 NaN。**差异只在进程边界。**

## 对测试的影响

`tests/` 里需要真引擎的用例已经标了 `slow`，它们是**每个用例一个引擎**
（pytest 的进程模型保证了这一点，只要不在同一个用例里建第二个）。

`conftest.py` 里如果有跨用例共享引擎的 fixture，那个引擎只能被同一个用例内
的多次操作复用；跨用例共享要谨慎。
