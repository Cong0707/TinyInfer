// FP8 e4m3 GEMV for decode (M=1)，激活的量化内联进 block —— 照 vLLM cutlass_scaled_mm 的
// 做法，把一次 act_quant 的「launch + 一趟全局往返」折进 GEMV 的 tile 里。
//
// 从 fp8_gemv_bs.cu 复制，唯一的改动是「激活从哪来」：这里吃原始 bf16 x，block 先用
// 32 个 warp 把 x 量化进 shared（算法与 act_quant.cu 逐位一致，函数逐字搬用），
// __syncthreads() 之后 32 个 warp 消费同一份量化结果。GEMV 主体（warps/lane 布局、
// TI_PREFETCH 软件流水、e4m3x4 硬件解码、scalar tail、warp 归约、acc += part*asc*wsc
// 的乘 scale 顺序）一字未动，所以输出与 act_quant + fp8_gemv_bs 两段式逐位相同。
//
// 为什么一个 block 算 32 行：量化一次 x 要读 2*k_dim 字节，block 只算 8 行时这份读取
// 就被放大 n_rows/8 次。一个 block 有 32 个 warp，阶段一里 warp 与 32 维组一一对应
// （g = warp; g < nb; g += 32），量化不额外吃线程，行数也就顺势定成 32 行/block：
// 放大压到 n_rows/32，而权重流量完全不变。代价：bf16 激活的全局读取是两段式（只读
// 一次）的 ceil(n_rows/32) 倍，但折算成字节只有 n_rows*k_dim/16，即权重流量的
// 1/16 = 6.25% —— 在权重主导的 decode 里可以忽略；换来的是每层少 5 次 launch 和
// 那份 5KB fp8 的全局写出/回读。
//
// 原文件说明（保留）：FP8 e4m3 GEMV for decode (M=1), block-scaled (32x32), bf16 out.
//
// Replaces the tilelang fp8_gemm for the shared expert: that kernel uses block_M=32, so
// at M=1 thirty-one of the thirty-two tile rows are padding. Measured 152 us for
// [2304, 5120] fp8 = 11.8 MB -> 78 GB/s, while the same shape at 4.25 bit reaches 316
// GB/s through Marlin. The gap is the tile shape, not the arithmetic.
//
// Layout (same idea as moe_gemv6.cu, which measures 246 GB/s on fp4):
//   block = 1024 = 32 output rows x 32 lanes
//   each lane owns 16 consecutive k (16 fp8 bytes, one LDG.E.128); a warp therefore
//   covers 512 consecutive k, so every fetch is full 128-byte transactions.
//   A scale group is 32 k, i.e. two lanes, so the group index is (lane >> 1).
//   Phase one uses the same 32 warps as 32 quant groups, so quantising costs no extra
//   threads: one warp owns one 32-element group, one shuffle reduce for its absmax.
//
// e8m0 needs no ldexpf: the byte IS the IEEE exponent field.
// e4m3 decodes through the hardware cvt.rn.f16x2.e4m3x2 (two values per instruction).
//
// 前置条件（与原两段式一致）：k_dim % 32 == 0（scale 组整组存在，act_quant 同样要求）；
// launch：grid = (n_rows + 31) / 32，block = 1024，shared = k_dim + (k_dim >> 5) 字节。
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

namespace {

__device__ __forceinline__ float e8m0(unsigned char b) {
    return __uint_as_float((unsigned int)b << 23);
}

// four packed e4m3 bytes -> four floats
__device__ __forceinline__ void e4m3x4(unsigned int raw, float* o) {
    const __half2_raw hr0 = __nv_cvt_fp8x2_to_halfraw2((unsigned short)(raw & 0xFFFFu), __NV_E4M3);
    const __half2_raw hr1 = __nv_cvt_fp8x2_to_halfraw2((unsigned short)(raw >> 16), __NV_E4M3);
    const float2 f0 = __half22float2(*reinterpret_cast<const __half2*>(&hr0));
    const float2 f1 = __half22float2(*reinterpret_cast<const __half2*>(&hr1));
    o[0] = f0.x; o[1] = f0.y; o[2] = f1.x; o[3] = f1.y;
}

// one packed e4m3 byte -> float, for the scalar tail path (K not a multiple of 512)
__device__ __forceinline__ float e4m3_one(unsigned char b) {
    float o;
    e4m3x4((unsigned int)b, &o);
    return o;
}

// ---- 以下四个函数逐字来自 act_quant.cu（量化算法一字未改）-------------------

// ceil(log2(x)) and 2^e by IEEE-754 bit manipulation, copied from the reference
// (official_kernel_ref.fast_log2_ceil / fast_pow2). This is NOT the same as
// ceilf(log2f(x)) / exp2f(e): the intrinsic has rounding error at exact powers of two,
// so a mantissa of 0 vs 1ulp flips the result by a whole binade. Measured: using the
// intrinsics changed the end-to-end decode output completely.
__device__ __forceinline__ int fast_log2_ceil(float x) {
    const unsigned int b = __float_as_uint(x);
    const int e = (int)((b >> 23) & 0xFFu);
    const unsigned int man = b & 0x7FFFFFu;
    return e - 127 + (man != 0u ? 1 : 0);
}

__device__ __forceinline__ float fast_pow2(int e) {
    return __uint_as_float((unsigned int)((e + 127) << 23));
}

__device__ __forceinline__ float round_scale(float amax) {
    // 448 is the largest finite e4m3 value; the reference multiplies by its
    // reciprocal rather than dividing, and that difference is one ulp.
    return fast_pow2(fast_log2_ceil(amax * (1.0f / 448.0f)));
}

__device__ __forceinline__ float warp_max(float v) {
    for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
    return v;
}

}  // namespace

extern "C" __global__ void __launch_bounds__(1024) fp8_gemv_inline_q(
    const __nv_bfloat16* __restrict__ x,    // [K]    bf16 activation（未量化）
    const unsigned char* __restrict__ w,    // [N, K] fp8 e4m3
    const unsigned char* __restrict__ ws,   // [N/32, K/32] e8m0 weight scales
    __nv_bfloat16* __restrict__ y,          // [N]
    int n_rows, int k_dim, float eps) {
    // 动态 shared：k_dim 字节 fp8 激活 + (k_dim >> 5) 字节 e8m0 scale。launch 时必须
    // 传 k_dim + (k_dim >> 5)，少了阶段二会读到别的 block 的 shared 之外。
    extern __shared__ __align__(16) unsigned char smem[];
    __nv_fp8_e4m3* aq = reinterpret_cast<__nv_fp8_e4m3*>(smem);
    unsigned char* asc = smem + k_dim;

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;      // 0 .. 31
    const int nb = k_dim >> 5;              // scale groups along k

    // ---- 阶段一：把 x 量化进 shared -------------------------------------------
    // 一个 warp 就是一组 32 个 k（32 个 lane 是这一组），absmax 是一次 shuffle
    // 归约，scale 一组算一次 —— 与 act_quant.cu 的调度完全相同，只把输出指向 shared。
    // 32 个 warp 每轮覆盖 32 组 = 1024 个 k，所以 K=5120 时第一轮就干完 31 组剩下
    // 的交给第二轮（g < nb 的循环上界，warp 内部所有 lane 迭代次数一致，无发散）。
    for (int g = warp; g < nb; g += 32) {
        const long long base = (long long)g * 32;
        const float v = __bfloat162float(x[base + lane]);

        float amax = warp_max(fabsf(v));
        // 参考实现的下界是 1e-4（调用方传 eps = 1e-4f）。这里再兜一层：万一 eps
        // 传 0，fast_log2_ceil(0) 会给出 2^-127 的 scale，下面就直接除零了。
        amax = fmaxf(amax, fmaxf(eps, 1e-4f));
        const float y_s = round_scale(amax);

        const float qv = fminf(fmaxf(v / y_s, -448.0f), 448.0f);
        aq[base + lane] = __nv_fp8_e4m3(qv);

        // One store per group, from lane 0. e8m0 is an exponent field only: the stored
        // byte IS the code, value = 2^(code - 127). y_s is already an exact power of two
        // (exp2(ceil(log2(.)))), so log2f(y_s) is an exact integer -- keep the same
        // lrintf form as act_quant.cu so the byte is bit-identical.
        if (lane == 0) {
            int code = (int)lrintf(log2f(y_s)) + 127;
            if (code < 0) code = 0;
            if (code > 255) code = 255;
            asc[g] = (unsigned char)code;
        }
    }
    // 整块同步：阶段二所有 warp 读同一份量化结果。这里不能提前 return，全部 1024
    // 个线程都必须到达这个 __syncthreads()。
    __syncthreads();

    // 阶段二的 a / as 指向 shared，其余与 fp8_gemv_bs.cu 逐字相同。
    const unsigned char* a = reinterpret_cast<const unsigned char*>(aq);
    const unsigned char* as = asc;

    // One warp handles one row, so the block covers 32 rows and the grid is
    // ceil(n_rows/32). Rows per block is no longer a free knob: the quant phase above
    // maps one warp per 32-element group, so a block must have 32 warps.
    const int n = blockIdx.x * 32 + (threadIdx.x >> 5);
    if (n >= n_rows) return;

    const unsigned char* wr = w + (long long)n * k_dim;
    const unsigned char* wsr = ws + (long long)(n >> 5) * nb;

    float acc = 0.f;
    // K need not be a multiple of 512: the loop rounds up and the last iteration handles
    // a short tail. That matters because two of the largest projections are NOT 512-
    // aligned -- wq_b is K=1280 (42 MB) and the shared expert's w2 is K=2304 -- and they
    // would otherwise fall through to the block_M=32 GEMM, the slowest thing left in the
    // decode step. A 16-byte vector load past the end of K would walk off the tensor, so
    // an incomplete lane group reads scalar instead.
    const int niter = (k_dim + 511) >> 9;   // 512 k per warp-iteration, rounded up
    // Software-pipelined loads. The loop body is load-then-use, so on a cold run -- which
    // is every decode, since each expert weight is a first touch out of HBM -- the full
    // HBM latency of each load sits exposed between iterations. That is worth a lot:
    // measured on the decode shapes the kernel delivers ~192 GB/s in situ against the
    // 323-395 GB/s it reaches when the data is warm. Issuing the next iteration's two
    // 16-byte vector loads before consuming the current pair lets them overlap with the
    // decode and the FMA of the current one.
    //
    // Nothing about the arithmetic moves: same iterations, same accumulation order, same
    // tail handling. Only the load is hoisted. (Here the "a" load is a shared LDS.128,
    // the "w" load is still the HBM one, so the pipeline keeps its point.)
    uint4 a_pre, w_pre;
    bool full_pre = false, valid_pre = false;
    int grp_pre = 0;
#define TI_PREFETCH(IT)                                                          \
    do {                                                                         \
        const int _kb = ((IT) << 9) + (lane << 4);                               \
        valid_pre = (_kb < k_dim);                                               \
        grp_pre = valid_pre ? (((IT) << 4) + (lane >> 1)) : 0;                   \
        full_pre = valid_pre && (_kb + 16) <= k_dim;                             \
        if (full_pre) {                                                          \
            a_pre = *(const uint4*)(a + _kb);                                    \
            w_pre = *(const uint4*)(wr + _kb);                                   \
        }                                                                        \
    } while (0)

    TI_PREFETCH(0);
    for (int it = 0; it < niter; ++it) {
        const uint4 araw = a_pre;
        const uint4 wraw = w_pre;
        const int kbase = (it << 9) + (lane << 4);
        // No continue: this loop contains a __shfl_xor_sync, and every __shfl*_sync in the
        // warp requires ALL 32 lanes to reach it. Skipping ahead on the tail lanes hung the
        // kernel outright (the test never returned). The guard is a bool instead, and the
        // tail lanes simply contribute zero.
        const bool valid = valid_pre;
        const int grp = grp_pre;
        const bool full = full_pre;
        if (it + 1 < niter) TI_PREFETCH(it + 1);

        float av[16];
        if (full) {
            const unsigned int* u = reinterpret_cast<const unsigned int*>(&araw);
#pragma unroll
            for (int i = 0; i < 4; ++i) e4m3x4(u[i], av + 4 * i);
        } else {
            for (int j = 0; j < 16; ++j) {
                const int kk = kbase + j;
                av[j] = (kk < k_dim) ? e4m3_one(a[kk]) : 0.f;
            }
        }
        const float asc_s = e8m0(as[grp]);
        const float wsc = e8m0(wsr[grp]);

        float part = 0.f;
        if (full) {
            const unsigned int* u = reinterpret_cast<const unsigned int*>(&wraw);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                float wv[4];
                e4m3x4(u[i], wv);
                part += av[4 * i] * wv[0] + av[4 * i + 1] * wv[1]
                      + av[4 * i + 2] * wv[2] + av[4 * i + 3] * wv[3];
            }
        } else if (valid) {
            for (int j = 0; j < 16; ++j) {
                const int kk = kbase + j;
                if (kk >= k_dim) break;
                part += av[j] * e4m3_one(wr[kk]);
            }
        }
        // Each lane applies its own group's scale, then the warp reduces. Pairing the two
        // lanes into (x1+x2)*s was tried to match tilelang's (sum of 32 k) * scale, and it
        // BROKE the shapes that used to be bit-identical: end to end went from
        // [989,43,4571,5958] to [989,19923,201,19923]. Whatever tilelang's gemm does
        // internally, it is not that, so this stays as the verified form.
        if (valid) acc += part * asc_s * wsc;
    }
    // Warp reduce. The groups are already summed in the same order as tilelang's
    // k loop within each lane; a fully serial k-order reduce was tried and is far
    // too slow (one shuffle per 32-k group, ~160 per call).
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) y[n] = __float2bfloat16(acc);
}
