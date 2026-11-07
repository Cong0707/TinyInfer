// fp8_gemv_dual: 一次 launch 算两个 fp8 GEMV，两个 GEMV 共用同一份已量化的激活。
//
// 结构完全照抄 fp8_gemv_bs.cu（block = warps 个 warp，一个 warp 一行，lane 扫 K，
// 每 lane 16 个 fp8 字节 = 一个 LDG.E.128，一个 scale 组是 32 个 k = 两个 lane，
// 所以组号是 lane >> 1）。e4m3 走硬件 cvt，e8m0 直接把字节当 IEEE 指数拿来用，
// 软件流水预取、K 不是 512 倍数时的 scalar tail、warp 内 shfl_down 归约，
// 一行都没有改动 —— 改的只有「这一行去哪个矩阵取权重」。
//
// 为什么合成一个 launch：
//   1. host 侧每 launch 一次要编组参数，实测约 19us；一个 decode step 里有 ~248 次
//      fp8 GEMV。两次合成一次，省下的是这个固定开销，不是带宽 —— 权重是两份，
//      照样要读两遍，这一点这个 kernel 省不掉。
//   2. 两个 grid 各有自己的尾浪。n1/8 和 n2/8 个 block 分别上 26 个 SM 时各留一次
//      尾巴，拼成一个 grid 只有一次尾巴。
//
// 为什么用 blockIdx 拼、而不是在 kernel 里把两个矩阵跑一遍：
//   归约是 warp 内的 __shfl_*_sync，全宽 shuffle 要求同一 warp 的 32 个 lane 走到
//   同一个矩阵的同一行。把「总行号 row -> (哪个矩阵, 矩阵内行号)」写在 warp 粒度上，
//   分界 n1 只可能落在两个 warp 之间，warp 内部不会分裂；写成 warp 内固定跑两遍
//   反而每个 warp 都要读两遍激活、两遍权重，还多一套累加器。
//
// 两个 kernel 只有最后一行 store 不同：fp8_gemv_dual 的两路输出都是 bf16，
// fp8_gemv_dual_q 的第二路是 fp32（有些调用方要 fp32 中间值，不要那一趟 bf16 舍入）。
// 计算部分共用 gemv_row，所以两条路径的累加顺序逐位一致。
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

namespace {

// e8m0 不需要 ldexpf：这个字节本身就是 IEEE 的指数域。
__device__ __forceinline__ float e8m0(unsigned char b) {
    return __uint_as_float((unsigned int)b << 23);
}

// 四个打包的 e4m3 字节 -> 四个 float（两条 cvt.rn.f16x2.e4m3x2）
__device__ __forceinline__ void e4m3x4(unsigned int raw, float* o) {
    const __half2_raw hr0 = __nv_cvt_fp8x2_to_halfraw2((unsigned short)(raw & 0xFFFFu), __NV_E4M3);
    const __half2_raw hr1 = __nv_cvt_fp8x2_to_halfraw2((unsigned short)(raw >> 16), __NV_E4M3);
    const float2 f0 = __half22float2(*reinterpret_cast<const __half2*>(&hr0));
    const float2 f1 = __half22float2(*reinterpret_cast<const __half2*>(&hr1));
    o[0] = f0.x; o[1] = f0.y; o[2] = f1.x; o[3] = f1.y;
}

// 单个 e4m3 字节 -> float，只给 scalar tail（K 不是 512 的倍数）用
__device__ __forceinline__ float e4m3_one(unsigned char b) {
    float o;
    e4m3x4((unsigned int)b, &o);
    return o;
}

// 一个 warp 算一行：y = sum_k aq[k] * e8m0(as[k/32]) * e4m3(wr[k]) * e8m0(wsr[k/32])
// wr / wsr 由调用方按行选好（w1 还是 w2），函数体与 fp8_gemv_bs 的内层逐行一致。
__device__ __forceinline__ float gemv_row(
    const unsigned char* __restrict__ a,    // [K]     fp8 e4m3 激活（两路共用）
    const unsigned char* __restrict__ as,   // [K/32]  e8m0 激活 scale
    const unsigned char* __restrict__ wr,   // [K]     本行权重 fp8 e4m3
    const unsigned char* __restrict__ wsr,  // [K/32]  本行所属 32 行组的 e8m0 权重 scale
    int k_dim, int lane) {
    float acc = 0.f;
    // K 不必是 512 的倍数：循环向上取整，最后一轮走短尾。wq_b 的 K=1280 和
    // shared expert w2 的 K=2304 都不是 512 对齐的。越过 K 末尾的 16 字节向量读
    // 会走出张量，所以不完整的 lane 组改读标量。
    const int niter = (k_dim + 511) >> 9;   // 一个 warp-迭代 512 个 k，向上取整
    // 软件流水：循环体是 load-then-use，冷跑时（decode 每次都冷，每个 expert 权重
    // 都是 HBM 首触）每轮 load 的完整延迟就裸露在两轮之间。把下一轮的两条 16 字节
    // 向量读提前发出去，让它和当前轮的译码 + FMA 重叠。算术一点没动：同样的迭代
    // 次数、同样的累加顺序、同样的尾部处理，只是把 load 提了上来。
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
        // 不能 continue：这个循环里有 __shfl_xor_sync，warp 内每个 __shfl*_sync 都要求
        // 全部 32 个 lane 到达。尾部 lane 提前跳过会把 kernel 直接挂死（测试一直不返回）。
        // 所以守卫换成 bool，尾部 lane 贡献 0。
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
        const float asc = e8m0(as[grp]);
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
        // 每个 lane 先乘自己那个组的 scale，再做 warp 归约。把两个 lane 配成
        // (x1+x2)*s 去贴合 tilelang「32 个 k 求和再乘 scale」的写法试过，破坏了原先
        // 逐位一致的形状（[989,43,4571,5958] -> [989,19923,201,19923]），所以保持这个
        // 已验证的形式。
        if (valid) acc += part * asc * wsc;
    }
#undef TI_PREFETCH
    // warp 归约。每个 lane 内部已经按 tilelang k 循环的顺序求和，全串行 k 序归约
    // 太慢（每 32 个 k 一次 shuffle，一次调用约 160 次）。
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    return acc;
}

}  // namespace

// 两路 bf16 输出。grid = ceil((n1 + n2) / warps)，blockDim 由 launcher 定
// （256 线程 = 8 行/块；warp 数只影响行怎么摊到 SM 上，算术与 block 大小无关，
// 所以它是填满尾浪的自由旋钮）。
extern "C" __global__ void fp8_gemv_dual(
    const unsigned char* __restrict__ a,    // [K]      fp8 e4m3 激活，两路共用
    const unsigned char* __restrict__ as,   // [K/32]   e8m0 激活 scale，两路共用
    const unsigned char* __restrict__ w1,   // [n1, K]  fp8 e4m3 第一路权重
    const unsigned char* __restrict__ ws1,  // [n1/32, K/32] e8m0 第一路权重 scale
    const unsigned char* __restrict__ w2,   // [n2, K]  fp8 e4m3 第二路权重
    const unsigned char* __restrict__ ws2,  // [n2/32, K/32] e8m0 第二路权重 scale
    __nv_bfloat16* __restrict__ y1,         // [n1] bf16
    __nv_bfloat16* __restrict__ y2,         // [n2] bf16
    int n1, int n2, int k_dim) {
    // 行数来自 launch，不是字面量：一个 warp 一行、归约在 warp 内完成，
    // 所以 block 的 warp 数只影响驻留，不影响结果。
    const int warps = blockDim.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * warps + (threadIdx.x >> 5);   // 拼起来的全局行号
    if (row >= n1 + n2) return;   // 整个 warp 一起退，后面的 shuffle 仍然整 warp 到达

    // row < n1 走第一路，否则走第二路的 (row - n1)。分界只落在 warp 之间，
    // warp 内部 32 个 lane 永远在同一个矩阵的同一行上，shuffle 语义不变。
    const bool first = (row < n1);
    const int n = first ? row : (row - n1);   // 在本矩阵内的行号
    const int nb = k_dim >> 5;                // 沿 k 的 scale 组数
    // 对应路数计数为 0 时，那个指针可以是 nullptr：三元式只求值被选中的那一支，
    // nullptr 不会进入指针算术。
    const unsigned char* wr = (first ? w1 : w2) + (long long)n * k_dim;
    const unsigned char* wsr = (first ? ws1 : ws2) + (long long)(n >> 5) * nb;

    const float acc = gemv_row(a, as, wr, wsr, k_dim, lane);
    if (lane == 0) {
        if (first) y1[n] = __float2bfloat16(acc);
        else       y2[n] = __float2bfloat16(acc);
    }
}

// 同一件事，第二路输出写成 fp32（调用方要 fp32 中间值时用，省掉一趟 bf16 舍入）。
// 除了最后一句 store，其余与 fp8_gemv_dual 完全相同，共用 gemv_row，
// 所以两版的累加顺序逐位一致，只是写出精度不同。
extern "C" __global__ void fp8_gemv_dual_q(
    const unsigned char* __restrict__ a,    // [K]
    const unsigned char* __restrict__ as,   // [K/32]
    const unsigned char* __restrict__ w1,   // [n1, K]
    const unsigned char* __restrict__ ws1,  // [n1/32, K/32]
    const unsigned char* __restrict__ w2,   // [n2, K]
    const unsigned char* __restrict__ ws2,  // [n2/32, K/32]
    __nv_bfloat16* __restrict__ y1,         // [n1] bf16
    float* __restrict__ y2,                 // [n2] fp32
    int n1, int n2, int k_dim) {
    const int warps = blockDim.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * warps + (threadIdx.x >> 5);
    if (row >= n1 + n2) return;

    const bool first = (row < n1);
    const int n = first ? row : (row - n1);
    const int nb = k_dim >> 5;
    const unsigned char* wr = (first ? w1 : w2) + (long long)n * k_dim;
    const unsigned char* wsr = (first ? ws1 : ws2) + (long long)(n >> 5) * nb;

    const float acc = gemv_row(a, as, wr, wsr, k_dim, lane);
    if (lane == 0) {
        if (first) y1[n] = __float2bfloat16(acc);
        else       y2[n] = acc;
    }
}
