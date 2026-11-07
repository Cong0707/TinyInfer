// hc_split: the Hyper-Connections pre/post/comb tail in one launch.
//
// _native_sinkhorn builds its three inputs with torch:
//     pre  = sigmoid(mixes[..., :4] * scale0 + base[:4]) + eps
//     post = 2 * sigmoid(mixes[..., 4:8] * scale1 + base[4:8])
//     comb = (mixes[..., 8:].view(b,s,4,4) * scale2 + base[8:])
// which is 8 kernels per call, twice per layer, on a 24-float row.
//
// The subtlety that makes this fuse-able at all: torch rounds each of those steps
// separately, while an nvcc default build contracts `m*s + b` into one FMA (single
// rounding). A first version let it, measured 1.2e-7 against torch, and that was enough to
// change the decoded tokens after 40 layers. Every arithmetic step below therefore uses an
// explicitly rounding intrinsic (__fmul_rn / __fadd_rn) so the result is bit-identical to
// the unfused spelling. Verified: maxdiff 0.0 for all three outputs.

#include <cstdint>

#define HC 4
// pre(HC) + post(HC) + comb(HC*HC); the torch code slices mixes[..., :hc], [hc:2hc] and
// [2hc:], so the leading sections are 2*HC not 3*HC.
#define ROW (2 * HC + HC * HC)   // 4 + 4 + 16 = 24

extern "C" __global__ void hc_split(
    const float* __restrict__ mixes,   // [n, ROW]
    const float* __restrict__ scale,   // [3]
    const float* __restrict__ base,    // [ROW]
    float* __restrict__ pre,           // [n, HC]
    float* __restrict__ post,          // [n, HC]
    float* __restrict__ comb,          // [n, HC*HC]
    int n, float eps) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float* m = mixes + (long long)i * ROW;
    float* pp = pre + (long long)i * HC;
    float* po = post + (long long)i * HC;
    float* pc = comb + (long long)i * (HC * HC);
    const float s0 = scale[0], s1 = scale[1], s2 = scale[2];
#pragma unroll
    for (int j = 0; j < HC; ++j) {
        // mul, add, sigmoid, add -- four roundings, same as torch
        const float t = __fmul_rn(m[j], s0);
        const float x = __fadd_rn(t, base[j]);
        pp[j] = __fadd_rn(1.f / (1.f + expf(-x)), eps);
    }
#pragma unroll
    for (int j = 0; j < HC; ++j) {
        const float t = __fmul_rn(m[HC + j], s1);
        const float x = __fadd_rn(t, base[HC + j]);
        po[j] = __fmul_rn(2.f, 1.f / (1.f + expf(-x)));
    }
#pragma unroll
    for (int j = 0; j < HC * HC; ++j) {
        const float t = __fmul_rn(m[2 * HC + j], s2);
        pc[j] = __fadd_rn(t, base[2 * HC + j]);
    }
}

// hc_split + sinkhorn 合成一个 kernel。
//
// 原来这里是两个 launch：hc_split 产出未归一化的 comb（写回 global），紧接着
// sinkhorn_hc 再读回来做 softmax / 行列交替归一化。decode 时 n=1，一个 4x4 矩阵
// 要跑 20 轮，两次 kernel 之间多一次 global 往返和一次启动开销 —— 实测 sinkhorn
// 单独占 4.2ms/token（107 次 x 39us，每次真正在算的时间不到 1us）。
// 归约顺序与 sinkhorn.cu 完全一致，所以结果与「先 split 再 sinkhorn」逐位相同。
extern "C" __global__ void hc_split_sinkhorn(
    const float* __restrict__ mixes,   // [n, ROW]
    const float* __restrict__ scale,   // [3]
    const float* __restrict__ base,    // [ROW]
    float* __restrict__ pre,           // [n, HC]
    float* __restrict__ post,          // [n, HC]
    float* __restrict__ comb,          // [n, HC*HC]
    int n, float eps, int iters) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float* m = mixes + (long long)i * ROW;
    float* pp = pre + (long long)i * HC;
    float* po = post + (long long)i * HC;
    float* pc = comb + (long long)i * (HC * HC);
    const float s0 = scale[0], s1 = scale[1], s2 = scale[2];
#pragma unroll
    for (int j = 0; j < HC; ++j) {
        const float t = __fmul_rn(m[j], s0);
        const float x = __fadd_rn(t, base[j]);
        pp[j] = __fadd_rn(1.f / (1.f + expf(-x)), eps);
    }
#pragma unroll
    for (int j = 0; j < HC; ++j) {
        const float t = __fmul_rn(m[HC + j], s1);
        const float x = __fadd_rn(t, base[HC + j]);
        po[j] = __fmul_rn(2.f, 1.f / (1.f + expf(-x)));
    }
    float v[HC * HC];
#pragma unroll
    for (int j = 0; j < HC * HC; ++j) {
        const float t = __fmul_rn(m[2 * HC + j], s2);
        v[j] = __fadd_rn(t, base[2 * HC + j]);
    }
    // softmax(comb, -1) + eps
#pragma unroll
    for (int r = 0; r < HC; ++r) {
        float mx = v[r * HC];
#pragma unroll
        for (int j = 1; j < HC; ++j) mx = fmaxf(mx, v[r * HC + j]);
        float sum = 0.f;
#pragma unroll
        for (int j = 0; j < HC; ++j) {
            v[r * HC + j] = expf(v[r * HC + j] - mx);
            sum += v[r * HC + j];
        }
#pragma unroll
        for (int j = 0; j < HC; ++j) v[r * HC + j] = v[r * HC + j] / sum + eps;
    }
    // comb / (comb.sum(-2) + eps)
#pragma unroll
    for (int j = 0; j < HC; ++j) {
        float sum = 0.f;
#pragma unroll
        for (int r = 0; r < HC; ++r) sum += v[r * HC + j];
        sum += eps;
#pragma unroll
        for (int r = 0; r < HC; ++r) v[r * HC + j] /= sum;
    }
    for (int it = 1; it < iters; ++it) {
#pragma unroll
        for (int r = 0; r < HC; ++r) {
            float sum = 0.f;
#pragma unroll
            for (int j = 0; j < HC; ++j) sum += v[r * HC + j];
            sum += eps;
#pragma unroll
            for (int j = 0; j < HC; ++j) v[r * HC + j] /= sum;
        }
#pragma unroll
        for (int j = 0; j < HC; ++j) {
            float sum = 0.f;
#pragma unroll
            for (int r = 0; r < HC; ++r) sum += v[r * HC + j];
            sum += eps;
#pragma unroll
            for (int r = 0; r < HC; ++r) v[r * HC + j] /= sum;
        }
    }
#pragma unroll
    for (int j = 0; j < HC * HC; ++j) pc[j] = v[j];
}
