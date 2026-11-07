// 手写 decode kernel（无 torch 头文件：nvcc -cubin，由 native.py 经 driver API 加载）。
//
// moe_fused_decode：把 decode 时一层 routed MoE 的四个 launch 收成一个。
// 原来的四段（见 kernels/cuda/moe_gemv6.cu 与 kernels/cuda/moe_act.cu 的头注释）：
//
//   moe_gemv6      gu[E, 2N] = gate_up(aq, w1/w3)                  fp8 e4m3 激活
//   moe_act_q6     a [E,  N] = silu(clamp(g)) * clamp(u) * w_exp[e]，并按 block=32 量化成 fp8
//   moe_gemv6      dn[E, 2D] = down(a, w2)                         fp8 激活，只消费平面 0
//   moe_acc6       y[d]      += sum_{e=0..E-1} dn[e][d]
//
// 本 kernel 的结构：一个 block 负责一段输出行 n，block 内先把**所有**专家的量化激活
// 算进自己的动态共享内存，再按专家 0..n_exp-1 的顺序把 down 的结果累加进 y。
//
// ============================ 一、算术逐位复现清单 ============================
// 文件里每一段算术都能对到参考实现上：
//
//  (1) GEMV：gemv_row / e4m3x8 / bf16x8 / e8m0 / kFp4Table 从 moe_gemv6.cu **逐字照抄**，
//      包括「每个 lane 连续 8 个 k、每 256 个 k 一步、32 宽 scale 组」的分块方式和
//      16/8/4/2/1 的 shuffle 归约树。gate 用 w1/s1、up 用 w3/s3、down 用 w2/s2，
//      行寻址 n*(k>>1)、scale 寻址 n*(k>>5) 与参考一致；权重基址 = arena + slot_ids[e]*blob_bytes + off。
//  (2) gate/up 的 bf16 回环：moe_gemv6 把 acc 用 bf16 写出、moe_act_q6 再 __bfloat162float
//      读回来，所以这里也必须先 __float2bfloat16 再读回（少这一步就多一位精度，silu 的输入就变了）。
//  (3) clamp：limit > 0 时 u 夹到 ±limit、g 只夹上界；与 moe_act.cu / moe_act_q6 同序同式。
//  (4) silu：s = g / (1.f + __expf(-g))（除法形式，两端都有限），再 (s*u)*w_exp[e] ——
//      乘法结合序与参考一致，路由权重**只在这里乘一次**。
//  (5) 量化：先 __float2bfloat16 回环（moe_act_q6 里 v 是 bf16 之后才参与 amax 与量化的），
//      然后 amax = warp_max(|v|)、amax = max(amax, 1e-4)、y_s = round_scale(amax)、
//      qv = clamp(v / y_s, ±448)、__nv_cvt_float_to_fp8(qv, SATFINITE, E4M3)；
//      scale 字节 = ex + 127，ex = fast_log2_ceil(amax * (1/448))。
//      fast_log2_ceil / fast_pow2 / round_scale / warp_max 从 act_quant.cu 逐字照抄；
//      moe_act_q6 里内联的 scale 位运算与此等价（它只是把这四行摊开了）。
//      注：act_quant 对 code 有 [0,255] 钳位，moe_act_q6 没有；这里跟 moe_act_q6 走，
//      因为 amax >= 1e-4 时 code 落在 (0,255) 内，钳位永不触发。
//  (6) down 的 bf16 回环：moe_gemv6 把 down 写成 bf16，moe_acc6 再读回来做 fp32 加法，
//      所以这里累加的是 __bfloat162float(__float2bfloat16(d))，不是裸的 d。
//  (7) down 之后**不再乘 w_exp**（moe_acc6 只做加法）；重复加权会让结果整体偏差。
//
// ============================ 二、浮点累加顺序 ============================
// 与 moe_acc6 **完全一致**。moe_acc6 的写法是
//
//     float acc = dst[i];  for (e = 0; e < n_experts; ++e) acc += float(src[e][i]);
//
// 本 kernel 的 down 段逐字对应：acc 初值是 float(y[n])，专家循环 0..n_exp-1 逐个 +=，
// 每个元素只由一个 lane 串行累加，没有原子加、没有跨 block 归约，所以顺序与 moe_acc6
// 的「顺序遍历专家」完全相同，多次运行也确定。
// 任务里允许的另一种「每个专家一组 block + 原子加」没有选：它的累加顺序由原子调度的
// 到达顺序决定，与 moe_acc6 不一致；而且它同样要为每个 block 重算一遍激活（见四），
// 代价与本方案相同，却把顺序弄丢了。
//
// ============================ 三、与参考实现的两处已知差异 ============================
//  (a) y 的类型：任务给定的签名是 __nv_bfloat16*，而当前 moe_acc6 的 dst 是 float*。
//      这里仍用 fp32 在寄存器里累加（acc），只在最后写回时舍入一次；但**读入**的初值是
//      bf16（原来的残差被舍入到 bf16 再进加法链）。若调用方的残差是 fp32，需要把本函数
//      的 y 改成 float*，并删掉 y 侧那两处 __float2bfloat16 / __bfloat162float 转换。
//  (b) aq 的布局：按任务给定的 [n_exp, k_dim] 逐专家索引。但现在的 moe_gemv.py 里
//      gate_up 传的是 act_stride=0（路由激活对所有专家是同一行），若调用方仍只有一行
//      共享激活，把下面的 aq + e*k_in / asc + e*(k_in>>5) 改成 aq / asc 即可，
//      算术完全相同，只是不重复读同一行。
//
// ============================ 四、代价：这一版为什么不是「更快」 ============================
// 签名里没有任何 scratch，所以每个 block 必须自己把全部 n_exp 个专家的激活算一遍才能做
// down（down 的归约维就是整条 act），gate_up 的权重流量因此被放大 G = gridDim.x 倍。
// 真实形状（6 专家 / k_in=5120 / n_gate_rows=2304）下一次 gate_up 是 70.8 MB，G 个 block
// 就是 G*70.8 MB；而这段 work 在四 launch 版本里由 1728 个 block 摊开（实测约 320 GB/s）。
// 单 block 的访存流水只有几十 GB/s，所以本 kernel 的耗时基本等于「一个 block 串行跑完
// 一次 gate_up」，G 只决定它被复制几份（有 L2 兜着），并不改变这个量级。
// 也就是说，这一版把 launch 开销换成了重算，用于精度/结构对照，不是性能路径。
// 真正能赢的形态是 vLLM fused_moe 的做法 —— 中间激活由调用方分配：
//   moe_fused_act (gate_up + clamp/silu/权重 + 量化 -> scratch)  1728 个 block，全并行
//   moe_fused_down(down + 按专家顺序累加 -> y)                    864 个 block，全并行
// 两段之间 Python 侧不需要再插别的 launch，顺序仍可与 moe_acc6 逐位一致。
// 要提速就在签名末尾补一个 const unsigned char* scratch（大小 n_exp*(k_mid + k_mid/32)）
// 并按上式拆成两个 kernel；本文件按任务给定的签名实现，没有加这个参数。
//
// ============================ 五、约束与启动配方 ============================
//   k_in % 256 == 0（gemv_row 每次跨 256 个 k）、k_mid % 256 == 0（down 的归约维同样走 256 步）
//   n_gate_rows 必须等于 k_mid（中间维度 N）；不相等时下面用 n_mid 取较小值做保护
//   blockDim.x % 32 == 0，建议 512（16 warp，与 moe_gemv6 的 block 形状一致）
//   动态共享内存 smem_bytes = n_exp*k_mid + n_exp*(k_mid>>5) + 2*k_mid   // 字节
//     （6 专家 / k_mid=2304 -> 18864 B 约 18.4 KB；8 专家同尺寸 -> 23616 B）
//     超过 48 KB 时要先 native.set_dynamic_smem("moe_fused", "moe_fused_decode", nbytes)
//   grid = (G, 1, 1)：每个 block 处理 ceil(n_down_rows / (G*16)) 行，G 在「并行度」和
//     「gate_up 重算份数」之间权衡（见四）。G 建议 16~32 起测。
//   需要 sm_89+（与 moe_gemv6.cu 对 fp8 cvt 指令的假设相同）

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

namespace {

// ---------------------------------------------------------------------------
// 以下五段逐字照抄 kernels/cuda/moe_gemv6.cu（fp4 查表 / e8m0 / e4m3 解码 / 行 GEMV）
// ---------------------------------------------------------------------------

__device__ __constant__ float kFp4Table[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

// e8m0 就是 IEEE 的指数域：2^(b-127) 的指数字节正是 b。
__device__ __forceinline__ float e8m0(const unsigned char b) {
    return __uint_as_float((unsigned int)b << 23);
}

// 8 个打包的 e4m3 字节 -> 8 个 float，四次硬件 cvt
__device__ __forceinline__ void e4m3x8(const unsigned char* p, float* o) {
#pragma unroll
    for (int q = 0; q < 2; ++q) {
        const unsigned int raw = *(const unsigned int*)(p + (q << 2));
        const __half2_raw hr0 = __nv_cvt_fp8x2_to_halfraw2(
            (unsigned short)(raw & 0xFFFFu), __NV_E4M3);
        const __half2_raw hr1 = __nv_cvt_fp8x2_to_halfraw2(
            (unsigned short)(raw >> 16), __NV_E4M3);
        const float2 f0 = __half22float2(*reinterpret_cast<const __half2*>(&hr0));
        const float2 f1 = __half22float2(*reinterpret_cast<const __half2*>(&hr1));
        o[q * 4 + 0] = f0.x;
        o[q * 4 + 1] = f0.y;
        o[q * 4 + 2] = f1.x;
        o[q * 4 + 3] = f1.y;
    }
}

// 8 个 bf16 -> 8 个 float。本 kernel 的两条 GEMV 都吃 fp8 激活，MODE 1 用不到，
// 但 gemv_row 是原样照抄的模板，所以这段也一并留着。
__device__ __forceinline__ void bf16x8(const unsigned char* p, float* o) {
    union {
        uint4 u;
        __nv_bfloat16 h[8];
    } cv;
    cv.u = *(const uint4*)p;
#pragma unroll
    for (int j = 0; j < 8; ++j) o[j] = __bfloat162float(cv.h[j]);
}

// MODE 0 = fp8 e4m3 激活，1 = bf16 激活
// 注意：a 也可能指向共享内存（本 kernel 的 down 段就是），泛型寻址下数值完全相同，
// 只是 ld.shared 会退化成 generic 的 ld。
template <int MODE>
__device__ __forceinline__ float gemv_row(const unsigned char* __restrict__ a,
                                          const unsigned char* __restrict__ as,
                                          const unsigned char* __restrict__ w,
                                          const unsigned char* __restrict__ ws,
                                          const int k_dim, const int lane,
                                          const float* lut) {
    const int niter = k_dim >> 8;   // 每步 256 个 k
    float acc = 0.f;
    for (int it = 0; it < niter; ++it) {
        const int kbase = (it << 8) + (lane << 3);      // 每个 lane 连续 8 个 k
        const unsigned int wpack = *(const unsigned int*)(w + (kbase >> 1));
        const int g = (it << 3) + (lane >> 2);          // kbase 所在的 32 宽 scale 组
        // bf16 激活带不了 block scale
        const float sc = (MODE == 0 ? e8m0(as[g]) : 1.f) * e8m0(ws[g]);
        float av[8];
        if (MODE == 0) {
            e4m3x8(a + kbase, av);
        } else {
            bf16x8(a + kbase * 2, av);
        }
        float part = 0.f;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            part += av[j] * lut[(wpack >> (j << 2)) & 0xF];
        }
        acc += part * sc;
    }
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    return acc;
}

// ---------------------------------------------------------------------------
// 以下四段逐字照抄 kernels/cuda/act_quant.cu（e8m0 的 scale 与 warp 归约）
// ---------------------------------------------------------------------------

// 用 IEEE-754 位运算做 ceil(log2(x)) 与 2^e。它不同于 ceilf(log2f(x)) / exp2f(e)：
// 在 2 的整数次幂附近，intrinsic 的舍入会让尾数 0 / 1ulp 翻转整整一个 binade。
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
    // 448 是 e4m3 的最大有限值；参考实现乘倒数而不是除，这两者差一个 ulp。
    return fast_pow2(fast_log2_ceil(amax * (1.0f / 448.0f)));
}

__device__ __forceinline__ float warp_max(float v) {
    for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
    return v;
}

}  // namespace

// ---------------------------------------------------------------------------
// 一个 block 负责一段输出行；block 内先把所有专家的量化激活算进共享内存，再按专家序累加。
// 动态共享内存布局（调用方按 smem_bytes 传入）：
//   [0, n_exp*k_mid)      a8 ：每个专家量化后的激活（fp8 e4m3，[e][k_mid]）
//   [.., + n_exp*nb)      as8：上面的 e8m0 scale 字节（[e][k_mid/32]）
//   [.., + 2*k_mid)       act：当前专家的 bf16 暂存，供 32 宽分组量化用
// ---------------------------------------------------------------------------
extern "C" __global__ void moe_fused_decode(
    const unsigned char* __restrict__ aq,        // [n_exp, k_dim] 已量化的激活 fp8 e4m3
    const unsigned char* __restrict__ asc,       // [n_exp, k_dim/32] e8m0
    const unsigned char* __restrict__ arena,     // 槽位大 buffer：[slot][blob_bytes]
    const int* __restrict__ slot_ids,            // [n_exp] 每个 active 专家的槽位编号
    const float* __restrict__ w_exp,             // [n_exp] 路由权重
    __nv_bfloat16* __restrict__ y,               // [dim] 累加输出（原地 +=）
    int n_exp, int n_gate_rows, int k_in, int n_down_rows, int k_mid,
    int blob_bytes,
    int w1_off, int s1_off, int w3_off, int s3_off,     // gate/up 的权重与 scale 偏移
    int w2_off, int s2_off,                              // down 的偏移
    float limit)
{
    extern __shared__ unsigned char smem[];
    const int nb = k_mid >> 5;                       // 每 32 维一组，共 nb 组
    unsigned char* a8 = smem;
    unsigned char* as8 = a8 + (long long)n_exp * k_mid;
    __nv_bfloat16* act = reinterpret_cast<__nv_bfloat16*>(as8 + (long long)n_exp * nb);

    // fp4 表放共享内存：用运行期 nibble 索引 __constant__ 会被 replay，shared 一次就出
    __shared__ float lut[16];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nwarp = blockDim.x >> 5;
    if (tid < 16) lut[tid] = kFp4Table[tid];
    __syncthreads();

    // gate 的行数就是中间维度；两者不相等时取小的那个，保证不越界写 act[]
    const int n_mid = (n_gate_rows < k_mid) ? n_gate_rows : k_mid;

    // ================= 阶段一：gemv(gate/up) -> clamp/silu/权重 -> fp8 量化 =================
    // 对应 moe_gemv6(gate_up) + moe_act_q6 两次 launch，逐个专家进行。
    for (int e = 0; e < n_exp; ++e) {
        // 槽位编号就是虚拟地址：arena + slot*blob_bytes + 层内固定偏移
        const long long eb = (long long)slot_ids[e] * (long long)blob_bytes;
        const unsigned char* w1 = arena + eb + w1_off;
        const unsigned char* s1 = arena + eb + s1_off;
        const unsigned char* w3 = arena + eb + w3_off;
        const unsigned char* s3 = arena + eb + s3_off;
        // 参考实现的 gate_up 传 act_stride=0：同一行激活喂所有专家。
        // 这里按任务给定的 [n_exp, k_dim] 布局逐专家索引；若上游只有一行共享激活，
        // 把下面两行改成 aq / asc 即可（算术完全相同）。
        const unsigned char* ae = aq + (long long)e * k_in;
        const unsigned char* ase = asc + (long long)e * (k_in >> 5);

        for (int n = warp; n < n_mid; n += nwarp) {
            const float g0 = gemv_row<0>(ae, ase, w1 + (long long)n * (k_in >> 1),
                                         s1 + (long long)n * (k_in >> 5), k_in, lane, lut);
            const float u0 = gemv_row<0>(ae, ase, w3 + (long long)n * (k_in >> 1),
                                         s3 + (long long)n * (k_in >> 5), k_in, lane, lut);
            if (lane == 0) {
                // (a) 复现 moe_gemv6 把结果写成 bf16、moe_act_q6 再从 bf16 读回的那次舍入
                float g = __bfloat162float(__float2bfloat16(g0));
                float u = __bfloat162float(__float2bfloat16(u0));
                // (b) clamp：limit <= 0 时不做
                if (limit > 0.f) {
                    u = fminf(fmaxf(u, -limit), limit);
                    g = fminf(g, limit);
                }
                // (c) silu(g) * u * w_exp[e]；路由权重只在这里乘**一次**
                const float s = g / (1.f + __expf(-g));
                // (d) 先舍入到 bf16：moe_act_q6 里 v 是 bf16 之后才参与 amax 和量化
                act[n] = __float2bfloat16(s * u * w_exp[e]);
            }
        }
        __syncthreads();

        // (e) 量化：一个 warp 负责一个 32 宽分组，与 moe_act_q6 / act_quant.cu 同调度
        for (int gi = warp; gi < nb; gi += nwarp) {
            const int i = (gi << 5) + lane;
            const float v = __bfloat162float(act[i]);
            float amax = warp_max(fabsf(v));
            amax = fmaxf(amax, 1e-4f);                 // 参考的下限
            const float y_s = round_scale(amax);       // 2^ceil(log2(amax/448))
            const float qv = fminf(fmaxf(v / y_s, -448.0f), 448.0f);
            a8[(long long)e * k_mid + i] =
                (unsigned char)__nv_cvt_float_to_fp8(qv, __NV_SATFINITE, __NV_E4M3);
            if (lane == 0) {
                // e8m0 只存指数：code = ex + 127，解码即 2^(code-127) = y_s
                const int ex = fast_log2_ceil(amax * (1.f / 448.f));
                as8[(long long)e * nb + gi] = (unsigned char)(ex + 127);
            }
        }
        __syncthreads();
    }

    // ===================== 阶段二：down + 按专家顺序累加进 y =====================
    // 对应 moe_gemv6(down, 平面 0) + moe_acc6 两次 launch。
    // acc 的初值与加法顺序与 moe_acc6 完全相同：float acc = dst[i]; for e: acc += ...
    for (int n = blockIdx.x * nwarp + warp; n < n_down_rows; n += gridDim.x * nwarp) {
        float acc = __bfloat162float(y[n]);            // moe_acc6 的 "acc = dst[i]"
        for (int e = 0; e < n_exp; ++e) {
            const long long eb = (long long)slot_ids[e] * (long long)blob_bytes;
            const unsigned char* w2 = arena + eb + w2_off + (long long)n * (k_mid >> 1);
            const unsigned char* s2 = arena + eb + s2_off + (long long)n * (k_mid >> 5);
            // a 与 as 指向本 block 自己算出来的共享内存激活（fp8 模式）
            const float d = gemv_row<0>(a8 + (long long)e * k_mid,
                                        as8 + (long long)e * nb, w2, s2, k_mid, lane, lut);
            // 复现 moe_gemv6 的 bf16 存储 + moe_acc6 的 bf16 读取：先舍入再加。
            // 注意：w_exp 已在阶段一乘过，这里**不再**乘（moe_acc6 只做加法）
            if (lane == 0) acc += __bfloat162float(__float2bfloat16(d));
        }
        if (lane == 0) y[n] = __float2bfloat16(acc);
    }
}