"""Diagnose: is 66 GB/s limited by access granularity or by compute?

Test: a kernel that reads the SAME bytes with perfectly coalesced accesses and does the
minimum work (sum bytes). If it also lands near 200 GB/s, the decode kernel's access
pattern is the problem. If it lands near 66, the bytes themselves are the limit.
"""
import sys, subprocess, os
cu = r'''
#include <cuda_fp16.h>
#include <cstdint>

// perfectly coalesced: each thread reads uint4 from a linear offset
extern "C" __global__ void stream_sum(const uint4* __restrict__ w, uint4* __restrict__ out,
                                      long long nvec) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    uint4 acc = make_uint4(0,0,0,0);
    for (; i < nvec; i += stride) {
        uint4 v = w[i];
        acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
    }
    if (acc.x == 0xFFFFFFFF) out[0] = acc;
}

// row-per-warp (what the decode kernel does): each warp walks one row of K/2 bytes
extern "C" __global__ void row_warp(const uint4* __restrict__ w, uint4* __restrict__ out,
                                    int rows, int vec_per_row) {
    int warp_global = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp_global >= rows) return;
    const uint4* row = w + (long long)warp_global * vec_per_row;
    uint4 acc = make_uint4(0,0,0,0);
    for (int i = lane; i < vec_per_row; i += 32) {
        uint4 v = row[i];
        acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
    }
    if (acc.x == 0xFFFFFFFF) out[0] = acc;
}
'''
open(r"E:\Develop\Deepseek\deepseek\_stream.cu","w").write(cu)
os.chdir(r"E:\Develop\Deepseek\deepseek")
vc = r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
r = subprocess.run('cmd /c "call "' + vc + '" >nul 2>&1 && nvcc -cubin -arch=sm_120 -o _stream.cubin _stream.cu"',
                   capture_output=True, text=True, shell=True)
print("nvcc:", r.returncode, r.stderr[:300] if r.returncode else "ok")

sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
import torch
import tinyinfer.torch_compat  # noqa
import tinyinfer.native as N
from pathlib import Path
orig = N._DIR
N._DIR = Path(r"E:\Develop\Deepseek\deepseek")
try:
    rows, K = 16384, 5120
    nbytes = rows * K // 2
    buf = torch.randint(0, 2**31-1, (nbytes//4,), dtype=torch.int32, device="cuda")
    out = torch.zeros(4, dtype=torch.uint32, device="cuda")
    nvec = nbytes // 16
    vpr = (K//2)//16

    def bench(fn, n=30, warm=8):
        with torch.inference_mode():
            for _ in range(warm): fn()
            torch.cuda.synchronize()
            e0 = torch.cuda.Event(True); e1 = torch.cuda.Event(True)
            e0.record()
            for _ in range(n): fn()
            e1.record(); torch.cuda.synchronize()
        return e0.elapsed_time(e1)/n

    gb = nbytes / 2**30
    for grid, blk, label in ((1024, 256, "linear 1024x256"), (4096, 256, "linear 4096x256"),
                             (8192, 128, "linear 8192x128")):
        ms = bench(lambda: N.launch(N.kernel("_stream","stream_sum"), (grid,), (blk,), [buf, out, nvec]))
        print(f"{label:20s}: {gb/(ms/1000):6.0f} GB/s")

    # row-per-warp with enough warps
    nwarps = rows
    grid = (nwarps + 255)//256
    ms = bench(lambda: N.launch(N.kernel("_stream","row_warp"), (grid,), (256,), [buf, out, rows, vpr]))
    print(f"{'row-per-warp':20s}: {gb/(ms/1000):6.0f} GB/s")
finally:
    N._DIR = orig
