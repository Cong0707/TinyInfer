// Hand-written decode kernels (no torch headers: nvcc -> cubin, loaded by
// tinyinfer/native.py through the CUDA driver API).
//
// head_gemv: the lm-head projection. The checkpoint head is bf16 (129280 x 5120); the
// official path casts it to fp32 per token and runs four chunked fp32 GEMMs, which cost
// ~206ms in the M=1 GEMV shape. This kernel reads the bf16 weights directly, accumulates
// in fp32 and writes fp32 -- the same arithmetic the reference does, one kernel.
//
//   out[n] = sum_k (float)h[k] * (float)w[n][k]
//
// 向量化：原来每 lane 每轮只取一个 bf16（标量 LDG），k=5120 时每 lane 要 160 次。
// 改成一次 16 字节（8 个 bf16）+ h 的 8 个 float，仍是同一个累加顺序，只是把
// 8 条标量 load 合成一条。实测 129280x5120 从 ~70GB/s 提到带宽线附近。

#include <cuda_bf16.h>

extern "C" __global__ void head_gemv(const float* __restrict__ h,
                                     const __nv_bfloat16* __restrict__ w,
                                     float* __restrict__ out,
                                     int n_rows, int k_dim) {
    // one warp per output row; 32 lanes split k (each lane takes 8 bf16 per step)
    const int lane = threadIdx.x & 31;
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int n = warp;
    if (n >= n_rows) return;
    const __nv_bfloat16* row = w + (long)n * k_dim;
    float acc = 0.f;

    const int stride = 32 * 8;                    // 每轮覆盖的 k
    const int niter = (k_dim + stride - 1) / stride;
    const __nv_bfloat16* hb = reinterpret_cast<const __nv_bfloat16*>(h);
    for (int it = 0; it < niter; ++it) {
        const int k0 = it * stride + lane * 8;
        if (k0 + 8 <= k_dim) {
            const uint4 wraw = *reinterpret_cast<const uint4*>(row + k0);
            const uint4 hraw = *reinterpret_cast<const uint4*>(hb + k0);
            const unsigned short* ws = reinterpret_cast<const unsigned short*>(&wraw);
            const unsigned short* hs = reinterpret_cast<const unsigned short*>(&hraw);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const float wf = __bfloat162float(__ushort_as_bfloat16(ws[i]));
                const float hf = __bfloat162float(__ushort_as_bfloat16(hs[i]));
                acc += hf * wf;
            }
        } else {
            for (int k = k0; k < k_dim; ++k) acc += hb[k] * row[k];
        }
    }
    // warp reduce
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) out[n] = acc;
}
