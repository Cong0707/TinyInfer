import torch
import vllm._C_stable_libtorch  # noqa
import vllm._moe_C_stable_libtorch  # noqa
from vllm.scalar_type import scalar_types
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    rand_marlin_weight_nvfp4_like,)

dev = "cuda"
def bench(fn, warmup=5, iters=30):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1000.0  # us

N, K = 16384, 5120
w = torch.randn(N, K, dtype=torch.bfloat16, device=dev)
ref, qw, sc, gs = rand_marlin_weight_nvfp4_like(w, 16)
print("qweight", tuple(qw.shape), qw.dtype, "bytes", qw.numel()*qw.element_size()/1e6, "MB")
print("scales", tuple(sc.shape), sc.dtype, "gscale", gs.item())
ws = torch.ops._C.marlin_make_workspace(N, dev)
print("workspace", tuple(ws.shape))

for M in (1, 8, 16, 32, 64):
    a = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
    fn = lambda: torch.ops.vllm.marlin_gemm(
        a, None, qw, None, sc, None, gs, None, ws,
        scalar_types.float4_e2m1f.id, M, N, K, False, False)
    t = bench(fn)
    wb = qw.numel()*qw.element_size()
    print(f"marlin_gemm fp4 M={M:3d} N={N} K={K}: {t:8.1f} us   {wb/(t*1e-6)/1e9:6.1f} GB/s")