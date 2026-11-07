// hc_pre 的投影：把超连接轴按 pre_mix 收起来。
//
// 原来这里是 _hc_pre_ref 的两三个算子：
//   torch.sum(pre_mix.unsqueeze(-1) * x.float(), dim=-2).to(x.dtype)
// 其中 .float() 把 [b,s,hc,d] 整块升到 fp32（再乘一遍、再 sum 一遍、再降回 bf16），
// 每层一次、40 层深。这个 kernel 在寄存器里用 fp32 累加，读 bf16 写 bf16，
// 算术与那条链一致（只是归约顺序变成顺序累加，hc 只有 4，顺序累加没有误差问题）。
#include <cuda_bf16.h>

extern "C" __global__ void hc_pre_collapse(
    const __nv_bfloat16* __restrict__ x,    // [rows, hc, d]
    const float* __restrict__ pre,          // [rows, hc]
    __nv_bfloat16* __restrict__ out,        // [rows, d]
    int rows, int hc, int d) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    const __nv_bfloat16* xr = x + (long long)row * hc * d;
    const float* pr = pre + (long long)row * hc;
    __nv_bfloat16* orow = out + (long long)row * d;
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        float acc = 0.f;
        for (int h = 0; h < hc; ++h) {
            acc += pr[h] * __bfloat162float(xr[(long long)h * d + i]);
        }
        orow[i] = __float2bfloat16(acc);
    }
}
