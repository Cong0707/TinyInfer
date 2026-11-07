"""Fix: keep 4-byte words but make consecutive lanes read consecutive 16-byte vectors.

Requests=1.15M, sectors=22.4M -> 19.6 sectors per request. A warp issuing one 4-byte load
per lane should touch only 4 sectors (128 bytes). 19.6 means the 32 lanes' addresses were
spread out, so each became its own sector.

The loop was  v = lane; v += 32  over 4-byte words, which IS contiguous across the warp...
unless the compiler split it. Check by measuring a version that reads uint4 explicitly with
lane consecutive.
"""
from pathlib import Path
p = Path(r"E:\Develop\Deepseek\tinyinfer\src\tinyinfer\native\gemv_fp4_e8m0.cu")
src = p.read_text(encoding="utf-8")
add = r'''
// gemv_fp4_fast4: one uint4 load per lane, lane-consecutive, so a warp covers 512
// contiguous bytes (16 sectors) in a single request.
//
// The profiler showed 19.6 sectors per load request for the 4-byte version, where a
// coalesced warp load of 128 bytes is 4 sectors. Reading uint4 with lane-consecutive
// offsets makes the request 512 bytes and the sector count exact.
extern "C" __global__ void gemv_fp4_fast4(
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
    const int nvec = khalf >> 4;               // 16-byte groups

    float p0 = 0.f, p1 = 0.f;
    for (int v = lane; v < nvec; v += 32) {
        const uint4 raw = *reinterpret_cast<const uint4*>(row + (long long)v * 16);
        const unsigned int words[4] = {raw.x, raw.y, raw.z, raw.w};
        const int k0 = v * 32;
#pragma unroll
        for (int wi = 0; wi < 4; ++wi) {
            const unsigned int word = words[wi];
            const int kb = k0 + wi * 8;
            p0 = fmaf(a[kb + 0], d4(word & 0xF), p0);
            p1 = fmaf(a[kb + 1], d4((word >> 4) & 0xF), p1);
            p0 = fmaf(a[kb + 2], d4((word >> 8) & 0xF), p0);
            p1 = fmaf(a[kb + 3], d4((word >> 12) & 0xF), p1);
            p0 = fmaf(a[kb + 4], d4((word >> 16) & 0xF), p0);
            p1 = fmaf(a[kb + 5], d4((word >> 20) & 0xF), p1);
            p0 = fmaf(a[kb + 6], d4((word >> 24) & 0xF), p0);
            p1 = fmaf(a[kb + 7], d4((word >> 28) & 0xF), p1);
        }
    }
    float acc = p0 + p1;
    float sc = 0.f;
    for (int kb = 0; kb < nb; ++kb) sc += as[kb];
    acc *= (nb > 0 ? sc / nb : 1.f);
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) y[n] = __float2bfloat16(acc);
}
'''
p.write_text(src + add, encoding="utf-8")
print("appended fast4")
