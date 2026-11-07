
import time, torch
import tinyinfer._C_stable_libtorch  # noqa
import tinyinfer._moe_C_stable_libtorch  # noqa
from tinyinfer import native

dev = "cuda"
x = torch.randn(1, 4096, dtype=torch.bfloat16, device=dev)
w = torch.randn(4096, dtype=torch.bfloat16, device=dev)
out = torch.empty_like(x)

def bench(fn, R=3000, w_=200):
    for _ in range(w_): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(R): fn()
    torch.cuda.synchronize()
    return (time.perf_counter()-t0)/R*1e6

# 1) torch.ops._C 调用（vLLM pyd）
us_ops = bench(lambda: torch.ops._C.rms_norm(out, x, w, 1e-6))
print(f"torch.ops._C.rms_norm      : {us_ops:.2f} us/call")

# 2) ctypes driver API 调用（我们 native.py 的方式）
try:
    fn = native._fn("rms_norm", "rms_norm_kernel") if hasattr(native, "_fn") else None
except Exception:
    fn = None
print(f"native._fn available       : {fn is not None}")

# 3) 纯 torch op 的最小开销（参照）
us_torch = bench(lambda: x.mul(1.0))
print(f"torch mul (reference)      : {us_torch:.2f} us/call")
