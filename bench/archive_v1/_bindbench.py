
import time, torch
import tinyinfer._C_stable_libtorch  # noqa
from tinyinfer import native

dev = "cuda"
D = 4096
x = torch.randn(1, D, dtype=torch.bfloat16, device=dev)
w = torch.randn(D, dtype=torch.bfloat16, device=dev)
out = torch.empty_like(x)

def bench(fn, R=5000, w_=500):
    for _ in range(w_): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(R): fn()
    torch.cuda.synchronize()
    return (time.perf_counter()-t0)/R*1e6

block = min(D // 8, 1024)
launcher = native.bind("vllm_rms_norm", "vllm_rms_norm_bf16", (1,), (block,), "TTTif")
launcher(out, x, w, D, 1e-6)
torch.cuda.synchronize()

u_old = bench(lambda: native.launch(native.kernel("vllm_rms_norm","vllm_rms_norm_bf16"), (1,), (block,), (out, x, w, D, 1e-6)))
u_new = bench(lambda: launcher(out, x, w, D, 1e-6))
u_ref = bench(lambda: torch.ops._C.rms_norm(out, x, w, 1e-6))
print(f"native.launch (generic) : {u_old:6.2f} us")
print(f"BoundLaunch (cached)    : {u_new:6.2f} us")
print(f"torch.ops._C            : {u_ref:6.2f} us")
print(f"ours vs _C: {u_ref/u_new:.2f}x")
