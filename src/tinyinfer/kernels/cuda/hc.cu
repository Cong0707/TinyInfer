// Deterministic Hyper-Connection kernels (no atomics: one thread per output element,
// sequential accumulation, so the result is bit-reproducible run to run).
//
// hc_pre :  y[m, d]        = sum_h pre[m, h] * x[m, h, d]
// hc_post:  y[m, a, b, d]  = post[m, a, b] * x[m, b, d]
//                          + sum_j comb[m, a, j] * resid[m, j, d]
// The second form is what the eager expression
//     post.unsqueeze(-1) * x.unsqueeze(-2) + sum(comb.unsqueeze(-1) * res.unsqueeze(-2), 2)
// broadcasts to: post[i,j] and comb[i,j] index the first two HC axes of the result, the
// value axis comes from x[j] and resid[j]. Replacing it with one kernel removes six
// launches per call (unsqueeze x2, mul x2, sum, cast) twice per layer.

#include <cuda_bf16.h>

extern "C" __global__ void hc_pre_f(const float* __restrict__ x,       // [M, hc, D]
                                    const float* __restrict__ pre,     // [M, hc]
                                    float* __restrict__ out,           // [M, D]
                                    int M, int hc, int D) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)M * D) return;
    int m = (int)(i / D);
    int d = (int)(i - (long)m * D);
    const float* xm = x + (long)m * hc * D + d;
    const float* pm = pre + (long)m * hc;
    float acc = 0.f;
    for (int h = 0; h < hc; ++h) acc += pm[h] * xm[(long)h * D];
    out[i] = acc;
}

extern "C" __global__ void hc_pre_bf(const __nv_bfloat16* __restrict__ x,
                                     const float* __restrict__ pre,
                                     __nv_bfloat16* __restrict__ out,
                                     int M, int hc, int D) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)M * D) return;
    int m = (int)(i / D);
    int d = (int)(i - (long)m * D);
    const __nv_bfloat16* xm = x + (long)m * hc * D + d;
    const float* pm = pre + (long)m * hc;
    float acc = 0.f;
    for (int h = 0; h < hc; ++h) acc += pm[h] * __bfloat162float(xm[(long)h * D]);
    out[i] = __float2bfloat16(acc);
}

// out is [M, hc, hc, D]: a and b are the two HC axes.
// out[m, o, d] = post[m, o] * x[m, o, d] + sum_i comb[m, i, o] * resid[m, i, d]
//
// This is the Hyper-Connection residual update, and it is the same arithmetic as the
// reference mhc_post_fwd:
//     x_out[n, o, h] = c[n, o] * d[n, h] + sum_i a[n, i, o] * b[n, i, h]
// Note the index order on comb: the sum runs over the *source* hyper-connection index i
// and o is the destination, which is why it is comb[m, i, o] and not comb[m, o, i].
// An earlier version of this kernel carried a fourth output dimension (b), treating post
// as [M, hc, hc] and reading x[m, b, d] -- that matches neither the caller's [M, hc]
// post nor the reference, and the resulting reshape threw on every call, so the fast
// path was dead and every token fell through to the torch expression in hc.py.
// x is the layer output BEFORE it is broadcast into hyper-connection space: it has shape
// [M, D], not [M, hc, D]. residual is the [M, hc, D] one. The caller's guard used to
// require residual.shape == x.shape, which can never hold for these two, so the fast path
// was unreachable on top of the index bug fixed here.
extern "C" __global__ void hc_post_f(const float* __restrict__ x,        // [M, D]
                                     const float* __restrict__ resid,    // [M, hc, D]
                                     const float* __restrict__ post,     // [M, hc]
                                     const float* __restrict__ comb,     // [M, hc, hc]
                                     float* __restrict__ out,            // [M, hc, D]
                                     int M, int hc, int D) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)M * hc * D) return;
    int d = (int)(i % D);
    long t = i / D;
    int o = (int)(t % hc);
    int m = (int)(t / hc);
    float acc = post[(long)m * hc + o] * x[(long)m * D + d];
    // Mix the hyper-connection states: out[m,o] += comb[m,i,o] * resid[m,i], summed over
    // the SOURCE index i. This matches the reference mhc_post_fwd
    // (a[n,i_mhci,i_mhco] * b[n,i_mhci,h]) and is what "residual update" has to mean --
    // the transpose reading (sum over the destination) leaves every o multiplying only its
    // own residual, which is not a mixing operation at all.
    for (int i_src = 0; i_src < hc; ++i_src)
        acc += comb[((long)m * hc + i_src) * hc + o] *
               resid[((long)m * hc + i_src) * D + d];
    out[i] = acc;
}

extern "C" __global__ void hc_post_bf(const __nv_bfloat16* __restrict__ x,        // [M, D]
                                      const __nv_bfloat16* __restrict__ resid,    // [M, hc, D]
                                      const float* __restrict__ post,             // [M, hc]
                                      const float* __restrict__ comb,             // [M, hc, hc]
                                      __nv_bfloat16* __restrict__ out,            // [M, hc, D]
                                      int M, int hc, int D) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)M * hc * D) return;
    int d = (int)(i % D);
    long t = i / D;
    int o = (int)(t % hc);
    int m = (int)(t / hc);
    float acc = post[(long)m * hc + o] * __bfloat162float(x[(long)m * D + d]);
    for (int i_src = 0; i_src < hc; ++i_src)
        acc += comb[((long)m * hc + i_src) * hc + o] *
               __bfloat162float(resid[((long)m * hc + i_src) * D + d]);
    out[i] = __float2bfloat16(acc);
}
// hc_mixes: row-normalised Hyper-Connection coefficient projection.
//
//   r   = rsqrt(mean_k(x[m,k]^2) + eps)
//   out[m, j] = r * sum_k x[m, k] * FN[j, k]
//
// The eager form is flatten + square + mean + add + rsqrt + linear + mul: seven launches
// for a 24x20480 projection, twice per layer. One block per row, one warp-wide reduction
// for the norm, and the projection rows are split across the block. No atomics: each
// output element is owned by one thread and accumulates sequentially, so the result is
// reproducible and matches the eager arithmetic order (fp32 accumulate).

#include <cuda_bf16.h>

#define HC_BLOCK 256

extern "C" __global__ void hc_mixes_bf(const __nv_bfloat16* __restrict__ x,  // [M, K]
                                       const float* __restrict__ fn,        // [J, K]
                                       float* __restrict__ out,             // [M, J]
                                       int M, int K, int J, float eps,
                                       int inv_k_is_pow2) {
    int m = blockIdx.x;
    if (m >= M) return;
    const __nv_bfloat16* xm = x + (long)m * K;

    __shared__ float s_norm;
    __shared__ float s_x[HC_BLOCK];
    int tid = threadIdx.x;

    // ---- row norm: sum of squares ----
    float part = 0.f;
    for (int k = tid; k < K; k += HC_BLOCK) {
        float v = __bfloat162float(xm[k]);
        part += v * v;
    }
    // warp tree, then one shared slot per warp
    for (int off = 16; off > 0; off >>= 1) part += __shfl_down_sync(0xffffffffu, part, off);
    __shared__ float s_warp[HC_BLOCK / 32];
    if ((tid & 31) == 0) s_warp[tid >> 5] = part;
    __syncthreads();
    if (tid == 0) {
        float s = 0.f;
        for (int w = 0; w < HC_BLOCK / 32; ++w) s += s_warp[w];
        s_norm = rsqrtf(s / (float)K + eps);
    }
    __syncthreads();
    float r = s_norm;

    // ---- projection: one output row per warp, K walked in HC_BLOCK-wide tiles ----
    for (int j = tid / 32; j < J; j += HC_BLOCK / 32) {
        const float* fnj = fn + (long)j * K;
        float acc = 0.f;
        // one warp per output row; the 32 lanes stride the whole K, then the warp tree
        // combines them. (Stepping k by the block width instead only covered 32 of every
        // 256 elements -- the projection came out 8x too small.)
        for (int k = tid & 31; k < K; k += 32) {
            acc += __bfloat162float(xm[k]) * fnj[k];
        }
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
        if ((tid & 31) == 0) out[(long)m * J + j] = acc * r;
    }
}

// fp32 activation variant (tier paths that carry fp32 h)
extern "C" __global__ void hc_mixes_f(const float* __restrict__ x,
                                      const float* __restrict__ fn,
                                      float* __restrict__ out,
                                      int M, int K, int J, float eps,
                                      int inv_k_is_pow2) {
    int m = blockIdx.x;
    if (m >= M) return;
    const float* xm = x + (long)m * K;
    __shared__ float s_norm;
    int tid = threadIdx.x;
    float part = 0.f;
    for (int k = tid; k < K; k += HC_BLOCK) {
        float v = xm[k];
        part += v * v;
    }
    for (int off = 16; off > 0; off >>= 1) part += __shfl_down_sync(0xffffffffu, part, off);
    __shared__ float s_warp[HC_BLOCK / 32];
    if ((tid & 31) == 0) s_warp[tid >> 5] = part;
    __syncthreads();
    if (tid == 0) {
        float s = 0.f;
        for (int w = 0; w < HC_BLOCK / 32; ++w) s += s_warp[w];
        s_norm = rsqrtf(s / (float)K + eps);
    }
    __syncthreads();
    float r = s_norm;
    for (int j = tid / 32; j < J; j += HC_BLOCK / 32) {
        const float* fnj = fn + (long)j * K;
        float acc = 0.f;
        for (int k = tid & 31; k < K; k += 32) {
            acc += xm[k] * fnj[k];
        }
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
        if ((tid & 31) == 0) out[(long)m * J + j] = acc * r;
    }
}

// hc_mixes 的并行版：一个 block 负责一个输出 j（grid = (J, M)）。
//
// 原 hc_mixes_bf 是 grid = (M,)，decode 时 M=1 —— 整层只有一个 block 在一块 SM 上
// 做 J*K 的乘加（24 x 20480），实测 11.6ms/token，是 GPU 侧第一大户。这里把 j 拆
// 到 grid.x，24 个 block 一起跑；行范数在每个 block 里重算一遍（K 次平方和对比
// 投影是零头），省掉一次额外的 kernel，也不必让所有 block 等一个共享值。
extern "C" __global__ void hc_mixes_bf_j(const __nv_bfloat16* __restrict__ x,  // [M, K]
                                         const float* __restrict__ fn,        // [J, K]
                                         float* __restrict__ out,             // [M, J]
                                         int M, int K, int J, float eps) {
    const int m = blockIdx.y;
    const int j = blockIdx.x;
    if (m >= M || j >= J) return;
    const __nv_bfloat16* xm = x + (long long)m * K;
    const float* f = fn + (long long)j * K;
    const int tid = threadIdx.x;

    __shared__ float s_warp[HC_BLOCK / 32];
    __shared__ float s_norm;

    float part = 0.f;
    for (int k = tid; k < K; k += HC_BLOCK) {
        const float v = __bfloat162float(xm[k]);
        part += v * v;
    }
    for (int off = 16; off > 0; off >>= 1) part += __shfl_down_sync(0xffffffffu, part, off);
    if ((tid & 31) == 0) s_warp[tid >> 5] = part;
    __syncthreads();
    if (tid == 0) {
        float t = 0.f;
        for (int w = 0; w < HC_BLOCK / 32; ++w) t += s_warp[w];
        s_norm = rsqrtf(t / (float)K + eps);
    }
    __syncthreads();

    float acc = 0.f;
    for (int k = tid; k < K; k += HC_BLOCK) {
        acc += __bfloat162float(xm[k]) * f[k];
    }
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    __syncthreads();
    if ((tid & 31) == 0) s_warp[tid >> 5] = acc;
    __syncthreads();
    if (tid == 0) {
        float t = 0.f;
        for (int w = 0; w < HC_BLOCK / 32; ++w) t += s_warp[w];
        out[(long long)m * J + j] = t * s_norm;
    }
}

// hc_mixes 与 hc_split_sinkhorn 合成一个 kernel。
//
// 原来是一次 hc_mixes_bf_j 把 24 个系数写回全局，紧接着 hc_split_sinkhorn 再读
// 回来做 sigmoid/softmax/Sinkhorn。中间那趟全局往返 + 一次 launch，在 decode
// （n=1）里完全没必要：24 个 float 就是这个 kernel 自己算出来的。
// 每一层调用两次（attention 一次、FFN 一次），40 层就是 160 次 launch/token。
extern "C" __global__ void hc_mixes_split_sinkhorn(
    const __nv_bfloat16* __restrict__ x,   // [M, K]
    const float* __restrict__ fn,          // [J, K]
    const float* __restrict__ scale,       // [3]
    const float* __restrict__ base,        // [J]
    float* __restrict__ pre,               // [M, HC]
    float* __restrict__ post,              // [M, HC]
    float* __restrict__ comb,              // [M, HC*HC]
    int M, int K, int J, float eps, int iters) {
    // 一个 block 负责一个 (m, j)：j 是 24 个系数中的一个，norm 每块重算。
    const int m = blockIdx.y;
    const int j = blockIdx.x;
    if (m >= M || j >= J) return;
    const __nv_bfloat16* xm = x + (long long)m * K;
    const int tid = threadIdx.x;

    __shared__ float s_warp[HC_BLOCK / 32];
    __shared__ float s_norm;
    __shared__ float s_mix[24];

    float part = 0.f;
    for (int k = tid; k < K; k += HC_BLOCK) {
        const float v = __bfloat162float(xm[k]);
        part += v * v;
    }
    for (int off = 16; off > 0; off >>= 1) part += __shfl_down_sync(0xffffffffu, part, off);
    if ((tid & 31) == 0) s_warp[tid >> 5] = part;
    __syncthreads();
    if (tid == 0) {
        float t = 0.f;
        for (int w = 0; w < HC_BLOCK / 32; ++w) t += s_warp[w];
        s_norm = rsqrtf(t / (float)K + eps);
    }
    __syncthreads();

    float acc = 0.f;
    for (int k = tid; k < K; k += HC_BLOCK) {
        acc += __bfloat162float(xm[k]) * fn[(long long)j * K + k];
    }
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    __syncthreads();
    if ((tid & 31) == 0) s_warp[tid >> 5] = acc;
    __syncthreads();
    if (tid == 0) {
        float t = 0.f;
        for (int w = 0; w < HC_BLOCK / 32; ++w) t += s_warp[w];
        s_mix[j] = t * s_norm;
    }
    __syncthreads();
    if (tid != 0 || j != 0) return;

    // ---- 以下与 hc_split_sinkhorn 的算术完全一致（同序、同 rounding）----
    float* pp = pre + (long long)m * HC;
    float* po = post + (long long)m * HC;
    float* pc = comb + (long long)m * (HC * HC);
    const float s0 = scale[0], s1 = scale[1], s2 = scale[2];
#pragma unroll
    for (int q = 0; q < HC; ++q) {
        const float t = __fmul_rn(s_mix[q], s0);
        const float xx = __fadd_rn(t, base[q]);
        pp[q] = __fadd_rn(1.f / (1.f + expf(-xx)), eps);
    }
#pragma unroll
    for (int q = 0; q < HC; ++q) {
        const float t = __fmul_rn(s_mix[HC + q], s1);
        const float xx = __fadd_rn(t, base[HC + q]);
        po[q] = __fmul_rn(2.f, 1.f / (1.f + expf(-xx)));
    }
    float v[HC * HC];
#pragma unroll
    for (int q = 0; q < HC * HC; ++q) {
        const float t = __fmul_rn(s_mix[2 * HC + q], s2);
        v[q] = __fadd_rn(t, base[2 * HC + q]);
    }
#pragma unroll
    for (int r = 0; r < HC; ++r) {
        const float mx = fmaxf(fmaxf(v[r * HC], v[r * HC + 1]),
                               fmaxf(v[r * HC + 2], v[r * HC + 3]));
        float sum = 0.f;
#pragma unroll
        for (int q = 0; q < HC; ++q) {
            v[r * HC + q] = expf(v[r * HC + q] - mx);
            sum += v[r * HC + q];
        }
#pragma unroll
        for (int q = 0; q < HC; ++q) v[r * HC + q] = v[r * HC + q] / sum + eps;
    }
#pragma unroll
    for (int q = 0; q < HC; ++q) {
        float sum = 0.f;
#pragma unroll
        for (int r = 0; r < HC; ++r) sum += v[r * HC + q];
        sum += eps;
#pragma unroll
        for (int r = 0; r < HC; ++r) v[r * HC + q] /= sum;
    }
    for (int it = 1; it < iters; ++it) {
#pragma unroll
        for (int r = 0; r < HC; ++r) {
            float sum = 0.f;
#pragma unroll
            for (int q = 0; q < HC; ++q) sum += v[r * HC + q];
            sum += eps;
#pragma unroll
            for (int q = 0; q < HC; ++q) v[r * HC + q] /= sum;
        }
#pragma unroll
        for (int q = 0; q < HC; ++q) {
            float sum = 0.f;
#pragma unroll
            for (int r = 0; r < HC; ++r) sum += v[r * HC + q];
            sum += eps;
#pragma unroll
            for (int r = 0; r < HC; ++r) v[r * HC + q] /= sum;
        }
    }
#pragma unroll
    for (int q = 0; q < HC * HC; ++q) pc[q] = v[q];
}
