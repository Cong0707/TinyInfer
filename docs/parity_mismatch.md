# 拼批与逐 token 的数值不一致 + 逐 token 崩溃（2026-11-08 14:17）

## 数据

```
N= 8  batch absmax=17.850   token absmax=14.625   maxdiff=4.63
N=16  batch absmax=17.673   token 路径 cuLaunchKernel failed: 1
```

两个独立问题：

### 1. 拼批与逐 token 的数值不一致（N=8）

`maxdiff=4.63`，相对误差 33%（`absmax` 只有 14-18）。
这远大于浮点噪声，说明**两条路径算的不是同一件事**。

argmax 相同（99919 vs 99919）纯属巧合 —— 那么大的误差，token 迟早会分叉。

### 2. 逐 token 路径在 N=16 时崩溃

```
cuLaunchKernel failed: 1     <- 参数不合法
[layer-fail] layer 0 streamed=False vram free=1208MB alloc=4088MB
```

显存充足（1.2GB 空闲），不是 OOM。是 kernel 参数问题 —— 很可能又是 grid
或参数个数的问题。

## 为什么这两个问题重要

**真实推理就是 prefill（多行）+ decode（单行）**。如果两条路径数值不一致，
那么 prefill 写进 KV 的东西和 decode 读出来的假设就不匹配，长上下文必然出错。

而 `N=16` 崩溃说明**逐 token 跑长序列会挂** —— 这正是「1M 上下文」最需要的路径。

## 已修复的相关 bug（本轮之前）

- `_MOE_FUSED["act"]` grid 含 `n_exp`，按名字缓存 → 已修
- `_MOE_FUSED["down"]` grid 含 `_rows`，按名字缓存 → 已修

修复后短输入 NaN 消失了，但**拼批与逐 token 的差异仍在**，
说明还有别的形状相关问题。

## 下一个探针

1. 对 N=8 逐层比较 batch 与 token 的 hidden，找出第一处不同的层。
2. 对 N=16 的 `cuLaunchKernel failed` 打开 `TI_GPUDBG=1` 拿堆栈，
   看是哪个 kernel、什么参数。

## 注意

`cuLaunchKernel failed: 1` 里的 1 是 `CUDA_ERROR_INVALID_VALUE`（不是 OOM 的 2）。
通常是 grid/block 维度为 0 或参数个数不匹配。
