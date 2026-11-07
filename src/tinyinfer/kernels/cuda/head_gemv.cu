// head_gemv: lm-head projection.  head 是 bf16 [129280, 5120] = 1.3GB，每 token 必读。
// M=1 时这是纯带宽 GEMV：理论 1.3GB / 864GB/s = 1.5ms。
//
// 实测瓶颈不在权重，而在激活 h 的重复读：原来每个 lane 每轮都从 global 取 8 个
// float。把 h 先搬进 shared（5120x4B = 20KB，每 block 一份），之后只读 shared。

#include <cuda_bf16.h>

__device__ __forceinline__ float bf16_to_f(unsigned short b) {
    return __uint_as_float(((unsigned int)b) << 16);
}

extern "C" __global__ void head_gemv(const float* __restrict__ h,
                                     const __nv_bfloat16* __restrict__ w,
                                     float* __restrict__ out,
                                     int n_rows, int k_dim) {
    extern __shared__ float hs[];
    const int tid = threadIdx.x;
    const int nthread = blockDim.x;
    for (int k = tid; k < k_dim; k += nthread) hs[k] = h[k];
    __syncthreads();

    const int lane = threadIdx.x & 31;
    const int warp = tid >> 5;
    const int row = blockIdx.x * (nthread >> 5) + warp;
    if (row >= n_rows) return;
    const __nv_bfloat16* wr = w + (long)row * k_dim;
    float acc = 0.f;

    const int stride = 32 * 8;
    const int niter = (k_dim + stride - 1) / stride;
    const unsigned short* ws = reinterpret_cast<const unsigned short*>(wr);
    for (int it = 0; it < niter; ++it) {
        const int k0 = it * stride + lane * 8;
        if (k0 + 8 <= k_dim) {
            const uint4 wraw = *reinterpret_cast<const uint4*>(wr + k0);
            const unsigned short* wv = reinterpret_cast<const unsigned short*>(&wraw);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                acc += hs[k0 + i] * bf16_to_f(wv[i]);
            }
        } else {
            for (int k = k0; k < k_dim; ++k) acc += hs[k] * bf16_to_f(ws[k]);
        }
    }
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) out[row] = acc;
}
