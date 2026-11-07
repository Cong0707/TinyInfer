// kv_norm_rope_quant —— MLA 每 token 的 KV 侧准备：把 kv_norm / apply_rope /
// act_quant(inplace) 三步加一次 cache 写入压成一个 kernel（wkv 的 GEMV 仍在外面）。
//
// 原来每层每 token 跑四步，本文件复现其中三步的算术：
//   1) kv_raw = wkv GEMV 的输出        [head_dim] bf16       （GEMV 不在本文件）
//   2) kv = rms_norm(kv_raw, norm_w)   [head_dim] bf16       （vllm_rms_norm / rms_norm_kernel）
//   3) kv = apply_rope(kv 的最后 64 维) [head_dim] bf16      （vllm_rotary_embedding_bf）
//   4) kv = act_quant(kv, inplace=True) [head_dim] bf16      （act_quant_dequant：fp8 量化再反量化）
//   5) cache[slot] = kv
//
// 这几步之间夹着两个 bf16 张量：rms_norm 交出去的是 bf16，RoPE 原地写回的也是 bf16，
// 量化器读到的就是那个已经取整过的行。所以下面两次 __float2bfloat16 不是多余的 ——
// 少任何一次，进量化器的数就和参考不是同一行 bf16，amax 可能整个翻一个 binade，
// 这个 token 的输出就直接变了。
//
// 一个 block 处理一个 token（单头）的 512 维，block 内映射固定：
//   线程 tid 负责 [4*tid, 4*tid+4) 这 4 个连续维；
//   一个 warp 128 维 = 4 个量化组，8 个 lane 对应一个 32 元素的组。
// 于是：读 kv_raw/norm_w、写 cache 都是每线程一次 8 字节的合并访问；RoPE 的每一对
// (2i, 2i+1) 都落在同一个线程的 4 维里（rope 区从 448 开始，是 4 的倍数），不用跨线程
// 倒数据；量化的 32 元素组又正好是 8 个 lane（每个 lane 贡献 4 维）。
//
// 不落 fp8、不落 scale：反量化后的 bf16 直接写进 cache。
//
// 参数（按顺序）：
//   kv_raw   输入   [head_dim] bf16，wkv 的 GEMV 输出，单 token 单头
//   norm_w   输入   [head_dim] bf16，kv_norm 的权重
//   cos_sin  输入   [rope_dim] fp32，当前 position 的 cos||sin（前 32 个 cos、后 32 个 sin）
//   cache    输入/输出 [num_slots, head_dim] bf16，按 slot 行写
//   slot     输入   int32，写哪一行（调用方一般用 position % window）
//   eps      输入   float，RMSNorm 的 eps，来自 cfg.rms_norm_eps
//                   （运行时配置，不能写死：deepseek 的 config 默认 1e-20，别的模型 1e-6）
//
// 启动：grid = (1, 1, 1)，block = (128, 1, 1)。线程映射写死 128，传错直接不做。
//
// 编码：本文件保存为 UTF-8 with BOM。中文 Windows 的 MSVC 把无 BOM 的 UTF-8 按 cp936
// 解，中文注释/字符串会吃字节，实测直接 57 个语法错误；带 BOM 后 nvcc -arch=sm_89 干净编过。
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

namespace {

// ---------------------------------------------------------------- 形状
constexpr int kHeadDim = 512;   // wkv 的输出宽度（单头）
constexpr int kRopeDim = 64;    // 只有最后 64 维做 RoPE
constexpr int kGroup = 32;      // 量化 block 大小（act_quant 的 block_size=32）
constexpr int kThreads = 128;   // 本 kernel 的线程映射写死 128
constexpr int kPerThread = kHeadDim / kThreads;  // 每线程 4 个连续维
constexpr int kRopeStart = kHeadDim - kRopeDim;  // 448
constexpr int kPairs = kPerThread / 2;           // 每线程 2 对
static_assert(kHeadDim == kThreads * kPerThread, "512 维要正好分给 128 个线程");
static_assert(kPerThread == 4, "8 个 lane 一个 32 元素组要求每 lane 4 维");
static_assert(kGroup == 8 * kPerThread, "同上：一个量化组 = 8 lane x 4 维");
static_assert(kRopeStart % kPerThread == 0, "rope 区必须按 4 维对齐，否则一对会跨线程");
static_assert(kRopeDim == 2 * 32, "cos/sin 各 32 项");

// ---------------------------------------------------------------- 量化常量
// 下面三个 device 函数从 act_quant.cu 原样照抄。它们不是等价变换：换成
// ceilf(log2f) / exp2f 在 2 的整数幂附近会差 1ulp，mantissa 的 0 vs 1ulp 就翻
// 一整个 binade，实测会直接改变 decode 输出。
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
    // 448 是 e4m3 能表示的最大有限值。参考乘的是倒数而不是除，这里必须一样。
    return fast_pow2(fast_log2_ceil(amax * (1.0f / 448.0f)));
}

// ---------------------------------------------------------------- 归约
// 一个 32 元素量化组的 absmax：本 lane 先取自己 4 维的最大，再在 8 个 lane 上
// xor 归约（4/2/1 只在这 8 个 lane 里跳，不会串到隔壁组）。act_quant.cu 是一个
// warp 恰好一个组、用 16..1 的整 warp xor 树；max 是精确的结合/交换运算，归约树
// 的形状不影响结果，只要参与归约的是同一批 32 个数。
__device__ __forceinline__ float group_max(float v) {
    for (int off = kPerThread; off > 0; off >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
    return v;
}

// warp 内 shfl_down 求和 + shared 合并各 warp 的部分和，然后广播 s[0]。
// 照 rms_norm_rope.cu 的 block_reduce_sum 一字不改：求和顺序换了，rstd 的末位就会
// 跟着变，而 rstd 会乘到整行 512 个数上。
__device__ __forceinline__ float block_reduce_sum(float v, float* s) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int nwarps = (blockDim.x + 31) >> 5;
    if (lane == 0) s[warp] = v;
    __syncthreads();
    if (warp == 0) {
        float t = (threadIdx.x < nwarps) ? s[threadIdx.x] : 0.f;
        for (int off = 16; off > 0; off >>= 1) t += __shfl_down_sync(0xffffffffu, t, off);
        if (threadIdx.x == 0) s[0] = t;
    }
    __syncthreads();
    return s[0];
}

}  // namespace

extern "C" __launch_bounds__(kThreads) __global__ void kv_norm_rope_quant(
    const __nv_bfloat16* __restrict__ kv_raw,   // [512]   wkv GEMV 的输出
    const __nv_bfloat16* __restrict__ norm_w,   // [512]   kv_norm 权重
    const float* __restrict__ cos_sin,          // [64]    cos[32] || sin[32]
    __nv_bfloat16* __restrict__ cache,          // [num_slots, 512]
    int slot,                                   // 写哪一行
    float eps) {                                // RMSNorm 的 eps
    // 线程映射写死 128；block 传错就不做，免得悄悄越界写坏 cache。
    if (blockDim.x != kThreads) return;

    const int tid = threadIdx.x;
    const int base = tid * kPerThread;  // 本线程负责的 4 个连续维的起点
    __shared__ float s_warp[32];        // 每个 warp 一个部分和（本 kernel 只用前 4 个）
    __shared__ float s_rstd;

    // --- 1) 读入 + fp32 平方和 -------------------------------------------------
    // 参考在 fp32 里算 sum(x^2)/d + eps 再 rsqrt。这里每线程把 [base, base+4) 按
    // 维序依次累加（单累加器、顺序加，nvcc 收缩成 FFMA，和参考的 variance += v*v
    // 同形），随后 warp 树 + shared 合并。
    // 8 字节读：base 是 4 的倍数，行起点按 512 bf16 = 1024B 对齐，所以 8B 对齐。
    const uint2 raw = *reinterpret_cast<const uint2*>(kv_raw + base);
    const float2 f01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw.x));
    const float2 f23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw.y));
    float v[kPerThread];
    v[0] = f01.x; v[1] = f01.y; v[2] = f23.x; v[3] = f23.y;

    float ss = 0.f;
#pragma unroll
    for (int j = 0; j < kPerThread; ++j) ss += v[j] * v[j];

    const float total = block_reduce_sum(ss, s_warp);
    if (tid == 0) s_rstd = rsqrtf(total / (float)kHeadDim + eps);
    __syncthreads();
    const float rstd = s_rstd;

    // --- 2) rms_norm：y = x * rstd * w，再取整到 bf16 ---------------------------
    // 参考管线里 rms_norm 交出来的是 bf16 张量，RoPE 读的就是它，所以这里必须取整。
    const uint2 wraw = *reinterpret_cast<const uint2*>(norm_w + base);
    const float2 w01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&wraw.x));
    const float2 w23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&wraw.y));
    const float w[kPerThread] = {w01.x, w01.y, w23.x, w23.y};
#pragma unroll
    for (int j = 0; j < kPerThread; ++j)
        v[j] = __bfloat162float(__float2bfloat16(v[j] * rstd * w[j]));

    // --- 3) RoPE：只动最后 rope_dim 维，交错布局 (2i, 2i+1) ---------------------
    // cos = cos_sin[i]，sin = cos_sin[rope_dim/2 + i]；
    // 公式和 vllm_rotary.cu 的 token_rope_bf 同形：x' = x*cos - y*sin，
    // y' = y*cos + x*sin。rope 区起点 448 是每线程 4 维的整数倍，一对不会跨线程。
    if (base >= kRopeStart) {
#pragma unroll
        for (int p = 0; p < kPairs; ++p) {
            const int i = (base - kRopeStart) / 2 + p;  // 第 i 对
            const float c = cos_sin[i];
            const float s = cos_sin[kRopeDim / 2 + i];
            const float a = v[2 * p];
            const float b = v[2 * p + 1];
            // apply_rope 是原地写回 bf16 张量的，所以这里也要取整一次。
            v[2 * p]     = __bfloat162float(__float2bfloat16(a * c - b * s));
            v[2 * p + 1] = __bfloat162float(__float2bfloat16(b * c + a * s));
        }
    }

    // --- 4) block-32 fp8 量化 + 反量化（逐位复现 act_quant_dequant） ------------
    // 8 个 lane 恰好是一个 32 元素组：组号 4*warp + lane/8，组起点 32*组号。
    // amax -> 下界 1e-4 -> y_s = 2^ceil(log2(amax/448)) -> q = clamp(v/y_s, ±448)
    // -> v' = float(fp8(q)) * y_s。
    float amax = 0.f;
#pragma unroll
    for (int j = 0; j < kPerThread; ++j) amax = fmaxf(amax, fabsf(v[j]));
    amax = group_max(amax);
    amax = fmaxf(amax, 1e-4f);            // 参考的下界，不能省
    const float y_s = round_scale(amax);  // y_s = 2^ceil(log2(amax/448))

    __nv_bfloat16 out_v[kPerThread];
#pragma unroll
    for (int j = 0; j < kPerThread; ++j) {
        const float qv = fminf(fmaxf(v[j] / y_s, -448.0f), 448.0f);
        const __nv_fp8_e4m3 q8 = __nv_fp8_e4m3(qv);    // fp8 e4m3 的 code
        out_v[j] = __float2bfloat16(float(q8) * y_s);  // 反量化成 bf16
    }

    // --- 5) 写 cache 的一行 ------------------------------------------------------
    // 4 个连续维 = 8 字节，一次 64 位 store；128 线程 x 8B = 1024B 正好一行，
    // 完全合并。行起点 slot*512 是 1024B 对齐，base 是 8B 对齐。
    const __nv_bfloat162 o01 = __halves2bfloat162(out_v[0], out_v[1]);
    const __nv_bfloat162 o23 = __halves2bfloat162(out_v[2], out_v[3]);
    uint2 out;
    out.x = *reinterpret_cast<const unsigned int*>(&o01);
    out.y = *reinterpret_cast<const unsigned int*>(&o23);
    __nv_bfloat16* dst = cache + (long long)slot * kHeadDim + base;
    *reinterpret_cast<uint2*>(dst) = out;
}
