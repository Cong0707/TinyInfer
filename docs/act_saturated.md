# rows=1 的 NaN：act 输出已饱和到 448（2026-11-08 14:05）

## 关键实测

在 `moe_fused_act` 之后检查它写出的 fp8 缓冲：

```
act: finite=True  absmax=448.0  numel=13824
moe out finite=False
```

**`absmax = 448.0` 正好是 fp8 e4m3 的饱和上限。** 这不是「偶尔有个大值」，
是**所有值都被顶到上限** —— 说明量化的是垃圾。

## 量化逻辑（kernel 第 304-313 行）

```c
float amax = warp_max(fabsf(v));       // 一个 warp（32 个 lane）内的最大值
amax = fmaxf(amax, 1e-4f);
const float y_s = round_scale(amax);   // 2^ceil(log2(amax/448))
const float qv = fminf(fmaxf(v / y_s, -448.0f), 448.0f);
```

`i = (gi << 5) + lane`，gi 覆盖 `[0, k_mid/32)`，i 覆盖 `[0, k_mid)`。
**这一步不涉及行维度**，所以不是「按行分组的错」。

## 那 448 从哪来

`qv = clamp(v / y_s, ±448)`。如果 `y_s` 比 `v` 小很多，所有值都会顶到 448。
而 `y_s = 2^ceil(log2(amax/448))`，正常应该让 `v/y_s` 落在 ±448 内。

**只有当 `amax` 远小于真实 `|v|` 时才会整体饱和。** 而 `amax` 是
`warp_max(fabsf(v))`，取的是同一批 32 个 v 的最大值 —— 理论上不可能小于真实值。

除非：**`act[]` 里的 v 本身已经饱和/异常**。

## 下一步（明确）

在第 303 行之前（`const float v = __bfloat162float(act[i]);`）打印 `act[]` 的
原始内容。如果 `act[]` 在量化前就已经是 ±448 量级，那问题在阶段一
（gate/up 的 GEMV + silu），而不是量化。

如果 `act[]` 正常但 `qv` 饱和，那就是 `y_s` 算错 —— 而 `y_s` 只依赖 `amax`，
那说明 **`warp_max` 在这个 block 配置下失效了**（比如 warp 数量与 block 大小不匹配）。

## 注意

`fa` 的 grid 是 `((nmid+31)//32, n_exp, 1)`，block 是 `(512,)` = 16 warp。
`moe_fused_act` 内部 `nb = k_mid / 32`。当 `k_mid = 2304` 时 `nb = 72`，
而 nwarp = 16 —— 每个 warp 要处理 4.5 个组。**循环 `gi += nwarp` 会处理 72/16
的非整数倍**，最后几个组可能被部分 warp 处理，`warp_max` 的结果就不完整。

这个疑点值得单独验证。
