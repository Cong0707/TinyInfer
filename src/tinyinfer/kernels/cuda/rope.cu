// rope_bf: apply_rope 的融合版 —— 一个 kernel 替掉
// float -> view_as_complex -> 复数乘 -> view_as_real -> flatten -> copy_（6 个）。
// 数学与 apply_rope 完全一致：把相邻两个元素看成复数乘 freqs（cos, sin），
// 内部用 fp32 算，写回 bf16（和 apply_rope 的 float 中间精度相同）。
#include <cuda_bf16.h>

extern "C" __global__ void rope_bf(
    const __nv_bfloat16* __restrict__ x,   // [rows, heads, dim]
    const float* __restrict__ freqs,       // [rows, dim/2 * 2] 交替 (cos, sin)
    __nv_bfloat16* __restrict__ out,
    int rows, int heads, int dim, int rot, int pos_stride) {
    // 调用约定：x/out 是 [rows * heads, dim]（head 已经并进第一维），freqs 是
    // [rot] 一组交错 (cos, sin)。pairs 用 rows*heads，调用方按同一个数量算 grid。
    // 每个线程处理一对 (2i, 2i+1)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int pairs = rows * heads * (rot >> 1);
    if (idx >= pairs) return;
    int r = idx / (heads * (rot >> 1));
    int rem = idx % (heads * (rot >> 1));
    int h = rem / (rot >> 1);
    int p = rem % (rot >> 1);
    int i = (dim - rot) + 2 * p;            // rope 作用在最后 rot 维
    long long base = ((long long)r * heads + h) * dim + i;
    float a = __bfloat162float(x[base]);
    float b = __bfloat162float(x[base + 1]);
    // freqs 只有一行（decode 时 Python 侧已把表切到当前这一步）：所有 row 共用
    // 同一组 (cos, sin)。原来写的是 r * (rot>>1)，行号一大于 0 就读到表外面去了。
    int f = p * 2;
    float c = freqs[f];
    float s = freqs[f + 1];
    out[base]     = __float2bfloat16(a * c - b * s);
    out[base + 1] = __float2bfloat16(a * s + b * c);
}

// 原地版：直接按 head 的 stride 读写父张量里那段 rope 切片。
//
// 非原地版要求 x 连续，而调用方传的都是 q[..., -rope_dim:] 这种切片（rope_dim 64
// 小于 head_dim 192，最后一维连续但 head 之间有间隔）。于是 reshape 要拷贝一份、
// kernel 算完再 copy_ 回原处 —— 每层两次（q 和 o 的 inverse）、每 token 八十次
// copy_，比 kernel 本身贵两个数量级。这里把 stride 交给 kernel，一个 launch 完成。
//
// 约定：x 指向切片的第一个元素（也就是 rope 段的起点），该维连续；heads 行之间
// 相隔 hstride 个元素。outer（batch*seq）必须为 1，decode 就是这种情况。
extern "C" __global__ void rope_inplace_bf(
    __nv_bfloat16* __restrict__ x,
    const float* __restrict__ freqs,   // [rot] 交错 (cos, sin)
    int heads, long long hstride, int rot, int inverse) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int pairs = heads * (rot >> 1);
    if (idx >= pairs) return;
    const int p = idx % (rot >> 1);
    const int h = idx / (rot >> 1);
    const long long base = (long long)h * hstride + 2 * p;
    const float a = __bfloat162float(x[base]);
    const float b = __bfloat162float(x[base + 1]);
    const float c = freqs[p * 2];
    float sn = freqs[p * 2 + 1];
    if (inverse) sn = -sn;
    x[base] = __float2bfloat16(a * c - b * sn);
    x[base + 1] = __float2bfloat16(a * sn + b * c);
}
