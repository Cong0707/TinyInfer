"""Can we use mma.sync on sm_120, and with which shapes?"""
import subprocess, tempfile, os
from pathlib import Path

src = r'''
#include <cuda_fp16.h>
#include <cstdio>

__global__ void probe_m16n8k16(const __half* a, const __half* b, float* c) {
#if __CUDA_ARCH__ >= 800
    unsigned int ra[4], rb[2];
    float rc[4] = {0,0,0,0};
    // mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(rc[0]), "+f"(rc[1]), "+f"(rc[2]), "+f"(rc[3])
        : "r"(ra[0]), "r"(ra[1]), "r"(ra[2]), "r"(ra[3]), "r"(rb[0]), "r"(rb[1]));
    c[threadIdx.x] = rc[0];
#endif
}

// bf16 variant (sm_80+)
__global__ void probe_bf16(const float* a, const float* b, float* c) {
#if __CUDA_ARCH__ >= 800
    unsigned int ra[4], rb[2];
    float rc[4] = {0,0,0,0};
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(rc[0]), "+f"(rc[1]), "+f"(rc[2]), "+f"(rc[3])
        : "r"(ra[0]), "r"(ra[1]), "r"(ra[2]), "r"(ra[3]), "r"(rb[0]), "r"(rb[1]));
    c[threadIdx.x] = rc[0];
#endif
}

// fp8 variant (sm_89+)
__global__ void probe_fp8(const unsigned int* a, const unsigned int* b, float* c) {
#if __CUDA_ARCH__ >= 890
    unsigned int ra[4], rb[2];
    float rc[4] = {0,0,0,0};
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(rc[0]), "+f"(rc[1]), "+f"(rc[2]), "+f"(rc[3])
        : "r"(ra[0]), "r"(ra[1]), "r"(ra[2]), "r"(ra[3]), "r"(rb[0]), "r"(rb[1]));
    c[threadIdx.x] = rc[0];
#endif
}

// int4 (s4) variant -- what marlin uses for fp4 weights
__global__ void probe_s4(const unsigned int* a, const unsigned int* b, int* c) {
#if __CUDA_ARCH__ >= 800
    unsigned int ra[2], rb[1];
    int rc[4] = {0,0,0,0};
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s4.s4.s32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
        : "+r"(rc[0]), "+r"(rc[1]), "+r"(rc[2]), "+r"(rc[3])
        : "r"(ra[0]), "r"(ra[1]), "r"(rb[0]));
    c[threadIdx.x] = rc[0];
#endif
}
'''
Path("E:\\Develop\\Deepseek\\deepseek\\_mma_probe.cu").write_text(src, encoding="utf-8")
print("wrote probe")
