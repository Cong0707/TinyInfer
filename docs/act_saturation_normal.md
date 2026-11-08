# act kernel 的量化饱和：代码看起来对（2026-11-08 14:06）

## `moe_fused_act` 的实际结构

```c
const int e = blockIdx.y;                    // 全局专家槽位 = row*topk + e
const int row0 = blockIdx.x * MOE_ACT_ROWS;  // 32 行一组
const unsigned char* ae = aq + e * k_in;     // [rows*topk, k] 展平

// 每个 warp 两行，写 sact[n - row0]
for (int r = 0; r < rpw; ++r) {
    const int n = row0 + warp + r * nwarp;
    if (n >= n_mid) continue;
    ... sact[n - row0] = __float2bfloat16(s * u * w_exp[e]);
}
__syncthreads();

// warp 0 做 32 宽量化
if (warp == 0) {
    const int gi = row0 >> 5;
    const int i = (gi << 5) + lane;
    float v = (i < n_mid) ? __bfloat162float(sact[lane]) : 0.f;
    float amax = warp_max(fabsf(v));
    ...
}
```

**索引都对**：`sact` 是 block 局部（0..31），`sact[lane]` 正确；
`aq + e*k_in` 与 Python 侧 `_aqx.view(_rows, _tk, _k)` 的 `[rows*topk, k]` 展平一致。

## 但 output 的 `absmax=448` 仍需解释

448 是 clamp 上限。`qv = clamp(v/y_s, ±448)`，只有 `|v/y_s| > 448` 才饱和。
而 `y_s = 2^ceil(log2(amax/448))`，所以 `v/y_s <= 448 * v/amax <= 448`，
**数学上不可能超过 448**（因为 amax >= |v|）。

所以 `absmax` 恰好等于 448 意味着：**有元素恰好等于 amax，且 amax/y_s = 448**，
这是 round_scale 的正常行为（它让最大值正好映射到 448）。

**`absmax=448` 本身不是错误！** fp8 e4m3 的量化本来就会把最大值放到 448。

## 这一轮的结论

我上一轮把 `absmax=448` 当成「饱和=垃圾」是**误判**。那是 fp8 量化的正常结果。

## 剩下什么

如果 act 的 fp8 输出正常，down 的输入正常，那 NaN 只能在：

1. `moe_fused_down` 的 `gemv_row` 在某个 `n` 上算错
2. `y[n]` 的累计（`acc += ...`）溢出
3. `w_exp[e]` 在 down 里被重复乘（注释说不再乘，实际呢）

## 建议的下一步

把 `moe_fused_down` 的输出单独截获（在 `fd(...)` 之后 `synchronize` 并检查 `y`），
看它是不是第一次就 NaN。如果是，问题在 down kernel；如果不是（第一次正常、
第二次开始 NaN），那是 `y` 的累计问题。
