
import sys
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
import torch
import tinyinfer._C_stable_libtorch  # noqa  (vendored, no vllm package needed)
import tinyinfer._moe_C_stable_libtorch  # noqa
from tinyinfer.gemm import fp4_linear
from tinyinfer.nn import rms_norm
from tinyinfer.moe import Expert

dev = torch.device("cuda")
def bench(fn, warmup=5, iters=30):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1000.0

print("=== vendored vLLM extension ===")
print("  marlin_gemm:", hasattr(torch.ops._C, "marlin_gemm"),
      "| rms_norm:", hasattr(torch.ops._C, "rms_norm"),
      "| grouped_topk:", hasattr(torch.ops._moe_C, "grouped_topk"))

N, K = 16384, 5120
wq = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=dev)
ws = torch.randint(120, 128, (N, K // 32), dtype=torch.uint8, device=dev)
x = torch.randn(1, K, dtype=torch.bfloat16, device=dev)
t = bench(lambda: fp4_linear(x, wq, ws))
print(f"  fp4 GEMV M=1: {t:.1f} us  {N*K//2/(t*1e-6)/1e9:.1f} GB/s")

w = torch.randn(K, dtype=torch.bfloat16, device=dev)
t = bench(lambda: rms_norm(x, w, 1e-6))
print(f"  rms_norm M=1 K=5120: {t:.2f} us")

def pack(r, k): return torch.randint(0, 256, (r, k // 2), dtype=torch.uint8, device=dev)
def sca(r, k): return torch.randint(120, 128, (r, k // 32), dtype=torch.uint8, device=dev)
e = Expert(None, None, None, swiglu_limit=0.0)
e.fp4 = (pack(N, K), sca(N, K), pack(K, N), sca(K, N), pack(N, K), sca(N, K))
y = e.forward_fp4(x)
torch.cuda.synchronize()
t = bench(lambda: e.forward_fp4(x))
print(f"  expert SwiGLU FFN M=1: {t:.1f} us  (finite={bool(torch.isfinite(y).all())})")
print("ALL OK")
