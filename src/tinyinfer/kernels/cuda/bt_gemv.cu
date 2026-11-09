// Batched GEMV：一次 launch 算 g 个独立的 [r, d] x [d] 。o = einsum("bsgd,grd->bsgr") 走
// cuBLAS 的 grouped batched gemm，实测 3.65 ms/token（每层一个 91us 的 kernel，占 GPU 时间 15%）。
// 这个形状是「每个 token 位置一个独立的矩阵」，batch 维在 g（128 组），所以拆开是
// 128 个互不相关的 GEMV —— 正好铺满一个 grid，不需要 grouped 的调度逻辑。
//
// 布局：w[g][r][d] bf16，x[g][d] bf16，y[g][r] bf16。累加用 fp32，和 cuBLAS 的
// bf16 GEMM 一样（tf32/fp32 累加），输出一次舍入。
#include <cuda_bf16.h>

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
// ---- 向量化版本 ----
//
// 上面那版每 lane 一次读 2 字节的 bf16。d=640 时每个 warp 要发 20 轮 2 字节
// 请求，内存事务极不经济 —— 实测 10.5MB 的权重跑 9.8ms，折合 1.07 GB/s，
// 而这块卡的显存带宽是 384 GB/s。
//
// 改成每次读 16 字节（8 个 bf16，用 uint4 承载），同样 32 lane 一轮覆盖 256
// 个元素，d=640 只要 2.5 轮。累加仍是 fp32，和上面完全一致；只是每个 lane
// 负责的元素集合变了，所以浮点加法顺序不同 —— 这在 fp32 累加下本来就是
// 不确定的（cuBLAS 也不保证），对比时要按容差比而不是按位比。
extern "C" __global__ void bt_gemv_bf16_vec(
    const __nv_bfloat16* __restrict__ x,   // [g, d]
    const __nv_bfloat16* __restrict__ w,   // [g, r, d]
    __nv_bfloat16* __restrict__ y,         // [g, r]
    int g, int r, int d) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int gi = blockIdx.y;
    const int row = blockIdx.x * (blockDim.x >> 5) + warp;
    if (row >= r) return;
    const __nv_bfloat16* xr = x + (long long)gi * d;
    const __nv_bfloat16* wr = w + ((long long)gi * r + row) * d;
    float acc = 0.f;
    const int vec = 8;                  // 8 x bf16 = 16 bytes
    const int nvec = d / vec;
    for (int v = lane; v < nvec; v += 32) {
        const uint4 wv = *reinterpret_cast<const uint4*>(wr + v * vec);
        const uint4 xv = *reinterpret_cast<const uint4*>(xr + v * vec);
        const __nv_bfloat162* wp = reinterpret_cast<const __nv_bfloat162*>(&wv);
        const __nv_bfloat162* xp = reinterpret_cast<const __nv_bfloat162*>(&xv);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 wf = __bfloat1622float2(wp[j]);
            const float2 xf = __bfloat1622float2(xp[j]);
            acc += wf.x * xf.x + wf.y * xf.y;
        }
    }
    // d 不是 vec 的整数倍时的余数（d=640 时没有，但不能假设）
    for (int k = nvec * vec + lane; k < d; k += 32) {
        acc += __bfloat162float(xr[k]) * __bfloat162float(wr[k]);
    }
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) y[(long long)gi * r + row] = __float2bfloat16(acc);
}
