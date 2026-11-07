
import os, torch, time
os.environ["CUDA_CACHE_DISABLE"] = "0"
import tinyinfer._C_stable_libtorch  # noqa
from tinyinfer import gemv_fp4_marlin

dev = "cuda"
N, K = 16384, 5120
torch.manual_seed(0)
w = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=dev)
s = torch.full((N, K // 32), 127, dtype=torch.uint8, device=dev)
x = torch.randn(1, K, dtype=torch.bfloat16, device=dev)

t0 = time.perf_counter()
gemv_fp4_marlin.gemv_fp4_fast(x, w, s)
torch.cuda.synchronize()
print(f"1st call (repack+JIT): {(time.perf_counter()-t0)*1e3:.1f} ms")

# repack 之后再测一次不带 repack 的
from tinyinfer.gemv_fp4_marlin import _prepared, _workspace, FP4_TYPE_ID, _ops
W = _prepared(w, s)
a = x.reshape(-1, W.size_k)
ops = _ops()
t0 = time.perf_counter()
for _ in range(20):
    ops.marlin_gemm(a, None, W.qweight, None, W.scales, None, None, None,
                    _workspace(x.device), FP4_TYPE_ID, a.size(0), W.size_n, W.size_k,
                    False, False, False)
torch.cuda.synchronize()
for _ in range(5):
    ops.marlin_gemm(a, None, W.qweight, None, W.scales, None, None, None,
                    _workspace(x.device), FP4_TYPE_ID, a.size(0), W.size_n, W.size_k,
                    False, False, False)
torch.cuda.synchronize()
t0 = time.perf_counter(); R = 100
for _ in range(R):
    ops.marlin_gemm(a, None, W.qweight, None, W.scales, None, None, None,
                    _workspace(x.device), FP4_TYPE_ID, a.size(0), W.size_n, W.size_k,
                    False, False, False)
torch.cuda.synchronize()
us = (time.perf_counter()-t0)/R*1e6
print(f"bare marlin_gemm: {us:.1f} us  {N*K/2/1e6/us*1e3:.1f} GB/s")
