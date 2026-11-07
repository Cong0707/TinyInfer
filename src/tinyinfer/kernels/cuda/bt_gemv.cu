// Batched GEMV：一次 launch 算 g 个独立的 [r, d] x [d] 。o = einsum("bsgd,grd->bsgr") 走
// cuBLAS 的 grouped batched gemm，实测 3.65 ms/token（每层一个 91us 的 kernel，占 GPU 时间 15%）。
// 这个形状是「每个 token 位置一个独立的矩阵」，batch 维在 g（128 组），所以拆开是
// 128 个互不相关的 GEMV —— 正好铺满一个 grid，不需要 grouped 的调度逻辑。
//
// 布局：w[g][r][d] bf16，x[g][d] bf16，y[g][r] bf16。累加用 fp32，和 cuBLAS 的
// bf16 GEMM 一样（tf32/fp32 累加），输出一次舍入。
extern "C" __global__ void bt_gemv_bf16(
    const __nv_bfloat16* __restrict__ x,   // [g, d]
    const __nv_bfloat16* __restrict__ w,   // [g, r, d]
    __nv_bfloat16* __restrict__ y,         // [g, r]
    int g, int r, int d) {
    // 一个 warp 算一行；blockDim = 256 -> 8 行/block，grid = (ceil(r/8), g)
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int gi = blockIdx.y;
    const int row = blockIdx.x * (blockDim.x >> 5) + warp;
    if (row >= r) return;
    const __nv_bfloat16* xr = x + (long long)gi * d;
    const __nv_bfloat16* wr = w + ((long long)gi * r + row) * d;
    float acc = 0.f;
    // 每 lane 处理 d/32 个元素，最后 warp 归约
    for (int k = lane; k < d; k += 32) {
        acc += __bfloat162float(xr[k]) * __bfloat162float(wr[k]);
    }
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) y[(long long)gi * r + row] = __float2bfloat16(acc);
}
