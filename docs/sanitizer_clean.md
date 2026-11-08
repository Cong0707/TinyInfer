# compute-sanitizer 结果：0 内存错误（2026-11-08 14:01）

## 实测

```
========= COMPUTE-SANITIZER
========= ERROR SUMMARY: 0 errors
```

对 `probe_min.py`（长短输入 + reset 的序列）跑 `--tool memcheck`：**没有任何
越界读写、没有未初始化读**。

## 这否定了我上一轮的两个判断

1. 「这是未定义行为」—— 不成立，内存访问全部合法。
2. 「需要 compute-sanitizer 才能定位」—— 跑了，它说没问题。

所以 NaN 是**数值计算产生的**，不是内存问题。

## 这个结论反而有价值

它排除了最大的一类可能性（15+ 项假设里最后的大类），而且排除得很干脆。
现在可以确定：

- 内存访问合法
- 输入有限
- 权重有限
- 但某一步的算术结果变成了 NaN

## 回到数值方向

`rms_norm_eps = 1e-20`。rms_norm 的实现是 `x / sqrt(mean(x^2) + eps)`。

如果 `mean(x^2)` 极小（比如整行都是 1e-30），分母接近 eps=1e-20，
但这仍然是个正常数，不会变 NaN。除非 **x 里有 inf**：`inf^2 = inf`，
`inf / inf = NaN`。

而 bf16 的 inf 来自哪里？`exp()` 溢出、除以零、或者**fp8 反量化时
scale = 0**（`x / 0` 或 `x * (1/0)`）。

## 下一个探针（明确）

在 MoE 之前的那次 `rms_norm(b.hc_pre(x1, apre), b.ffn_norm, eps)` 里：

```python
v = b.hc_pre(x1, apre)
print("pre-norm finite:", torch.isfinite(v).all(), "absmax:", v.abs().max())
print("mean(x^2) min:", (v.float()**2).mean(-1).min())
out = rms_norm(v, b.ffn_norm, eps)
print("post-norm finite:", torch.isfinite(out).all())
```

如果 `pre-norm` 就含 inf，问题在 hc_pre；如果 mean(x^2) 有 0，问题在归一化。

## 注意

`_forward_gpu_only` 里也有归一化/量化步骤。上一轮的 fp8 路径缺失修复之后，
这条路径可能真的走到了新的分支 —— 值得重测一次 `probe_min.py`（已经测了，
结果仍是 NaN，所以不是那条路径）。
