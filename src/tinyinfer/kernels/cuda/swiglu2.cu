// Shared expert 的 SwiGLU：两个 linear 的输出直接进一个 kernel。
//
// 原来（Expert.forward）是：
//   gate = w1(x).float(); up = w3(x).float()
//   up = clamp(up, -limit, limit); gate = clamp(gate, max=limit)
//   y = silu(gate) * up
// 一共 7 个算子。QuantLinear 的输出本来就是 bf16，.float() 只是把同一批数值换个
// dtype，所以这里直接读 bf16、在寄存器里用 fp32 算，结果逐 bit 相同，kernel 数从
// 七个变成一个（vLLM 的 SiluAndMul 也是这个形状：bf16 进、fp32 算、bf16 出）。
#include <cuda_bf16.h>

extern "C" __global__ void swiglu2_bf(
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    __nv_bfloat16* __restrict__ out,
    int n, float limit) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = __bfloat162float(gate[i]);
    float u = __bfloat162float(up[i]);
    if (limit > 0.f) {
        g = fminf(g, limit);
        u = fminf(fmaxf(u, -limit), limit);
    }
    const float s = g / (1.f + __expf(-g));   // silu
    out[i] = __float2bfloat16(s * u);
}
