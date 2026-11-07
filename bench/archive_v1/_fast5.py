"""fast4 is slower with the SAME sector count and FEWER instructions.

So the difference is scheduling: fast4's unrolled 32-FMA body per iteration has a long
dependent chain through p0/p1, while fast's 8-FMA body lets the warp retire an iteration
and overlap the next load.

Test: keep fast's loop shape (one word per iteration) but split into 4 accumulators, and
let the compiler pipeline two words per iteration.
"""
from pathlib import Path
p = Path(r"E:\Develop\Deepseek\tinyinfer\src\tinyinfer\native\gemv_fp4_e8m0.cu")
src = p.read_text(encoding="utf-8")
add = r'''
// gemv_fp4_fast5: fast's per-iteration shape, but two words and four accumulators.
//
// fast4 (one uint4 = 32 nibbles per iteration) measured 3.7x SLOWER than fast despite
// identical L1 sector counts and fewer instructions, so the limit is not traffic: the
// 32-FMA unrolled body forms a dependency chain that the warp cannot overlap with the
// next load. Here each iteration reads two words (8 nibbles each) into four independent
// accumulators, giving the scheduler four chains and a shorter body than fast4.
extern "C" __global__ void gemv_fp4_fast5(
    const float* __restrict__ a,
    const float* __restrict__ as,
    const unsigned char* __restrict__ w,
    const unsigned char* __restrict__ ws,
    __nv_bfloat16* __restrict__ y,
    int n_rows, int k_dim) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int n = blockIdx.x * 8 + warp;
    if (n >= n_rows) return;

    const int nb = k_dim >> 5;
    const long long khalf = k_dim >> 1;
    const unsigned char* row = w + (long long)n * khalf;
    const int nword = khalf >> 2;                // 4-byte words per row

    float p0 = 0.f, p1 = 0.f, p2 = 0.f, p3 = 0.f;
    for (int v = lane; v < nword; v += 64) {
        const int v2 = v + 32;
        unsigned int w0, w1;
        memcpy(&w0, row + (long long)v * 4, 4);
        if (v2 < nword) memcpy(&w1, row + (long long)v2 * 4, 4); else w1 = 0;
        const int k0 = v * 8, k1 = v2 * 8;
        p0 = fmaf(a[k0 + 0], d4f(w0 & 0xF), p0);
        p1 = fmaf(a[k0 + 1], d4f((w0 >> 4) & 0xF), p1);
        p2 = fmaf(a[k0 + 2], d4f((w0 >> 8) & 0xF), p2);
        p3 = fmaf(a[k0 + 3], d4f((w0 >> 12) & 0xF), p3);
        p0 = fmaf(a[k0 + 4], d4f((w0 >> 16) & 0xF), p0);
        p1 = fmaf(a[k0 + 5], d4f((w0 >> 20) & 0xF), p1);
        p2 = fmaf(a[k0 + 6], d4f((w0 >> 24) & 0xF), p2);
        p3 = fmaf(a[k0 + 7], d4f((w0 >> 28) & 0xF), p3);
        if (v2 < nword) {
            p0 = fmaf(a[k1 + 0], d4f(w1 & 0xF), p0);
            p1 = fmaf(a[k1 + 1], d4f((w1 >> 4) & 0xF), p1);
            p2 = fmaf(a[k1 + 2], d4f((w1 >> 8) & 0xF), p2);
            p3 = fmaf(a[k1 + 3], d4f((w1 >> 12) & 0xF), p3);
            p0 = fmaf(a[k1 + 4], d4f((w1 >> 16) & 0xF), p0);
            p1 = fmaf(a[k1 + 5], d4f((w1 >> 20) & 0xF), p1);
            p2 = fmaf(a[k1 + 6], d4f((w1 >> 24) & 0xF), p2);
            p3 = fmaf(a[k1 + 7], d4f((w1 >> 28) & 0xF), p3);
        }
    }
    float acc = (p0 + p1) + (p2 + p3);
    float sc = 0.f;
    for (int kb = 0; kb < nb; ++kb) sc += as[kb];
    acc *= (nb > 0 ? sc / nb : 1.f);
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) y[n] = __float2bfloat16(acc);
}
'''
p.write_text(src + add, encoding="utf-8")
print("appended fast5")
