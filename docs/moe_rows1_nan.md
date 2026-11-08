# 锁定了：MoE kernel 在 `_rows == 1` 时算出 NaN（2026-11-08 14:03）

## 证据链完整

调用 `b.ffn.forward(hn2)`（单行），所有输入都已验证：

```
hn2            finite=True  absmax=0.6523
gate weights   (1,6) min=0.207 max=0.319 sum=1.500  finite=True
gate idx       [251, 19, 153, 250, 54, 52]          合法专家 id
moe.forward    finite=False absmax=nan
_gpu_fail      None                    <- 没有异常，GPU 路径成功执行
```

**没有异常、没有回退、输入全对，kernel 就是算出了 NaN。**

## 和 `_rows` 的关系

| rows | 结果 |
| --- | --- |
| 11 | finite |
| 4 | finite（在干净引擎下）|
| 1 | **NaN** |

## 相关 kernel 与它们的 row 维度

```python
fa = BoundLaunch(kernel("moe_fused", "moe_fused_act"),
                 ((nmid+31)//32, n_exp, 1), (512,), "TTTTTOOiiiiiiiif")
fd = BoundLaunch(kernel("moe_fused", "moe_fused_down"),
                 ((dim+15)//16, _rows), (512,), "TTTTOiiiiiiii")
```

- `fa` 的 grid 是 `(nmid/32, n_exp)` —— 与 `_rows` 无关（它按 [expert] 铺）
- `fd` 的 grid 是 `(dim/16, _rows)` —— **y 维度就是行数**

`_rows == 1` 时 `fd` 的 grid 是 `(320, 1)`，只有一个 block 行。

## 最可疑的点

`moe_fused_down` 的 kernel 里 `row_d = blockIdx.y`，当 `gridDim.y == 1` 时
只算第 0 行 —— 这是正常的。但：

**`moe_fused_act` 写的是 `a[n_exp][nmid]`，`moe_fused_down` 读的是同一个 buffer。**
如果 `act` 在某些 `n_exp` 下没写满（比如它的 grid 是 `n_exp` 但只填了部分），
`down` 会读到未初始化的 `scr`。

而 `scr` 是 `_scratch(n_exp * (nmid + (nmid>>5)))` —— **只增不减的共享 buffer**。
前面那次 11 行 prefill 用的 `n_exp=66`，scratch 被填满；现在 `n_exp=6`，
只写了前 6 个专家的位置，**但 `down` 可能按旧的 stride 读**。

## 具体要查

`moe_fused_down` 的 kernel 参数里有 `nmid` 和 `n_exp`。如果它用 `nmid` 做 stride
去索引 `a`，而 `a` 的实际布局是 `[n_exp][nmid]`，那 stride 是 `nmid`（对）。

但 **`scr` 的切片**：

```python
fa(scr[:n_exp * nmid], scr[n_exp * nmid:], ...)   # act 写这两段
fd(scr[:n_exp * nmid], scr[n_exp * nmid:], ...)   # down 读这两段
```

两处切片一致。所以布局没问题。

## 下一步

在 `_forward_gpu_only` 里，`fd(...)` 之后立刻 `zeros_` 检查 `y` 的内容，
以及检查 `scr` 在 `fa` 之后是否 finite：

```python
fa(...)
torch.cuda.synchronize()
print("after act: scr finite", torch.isfinite(scr[:n_exp*nmid].view(torch.float8_e4m3fn).float()).all())
```

如果 `act` 的输出已经含 NaN/inf，问题在 act；否则在 down。
