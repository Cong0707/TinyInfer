
import time, torch
import tinyinfer._C_stable_libtorch  # noqa
from tinyinfer import native

dev = "cuda"
D = 4096
x = torch.randn(1, D, dtype=torch.bfloat16, device=dev)
w = torch.randn(D, dtype=torch.bfloat16, device=dev)
out = torch.empty_like(x)

def bench(fn, R=3000, w_=300):
    for _ in range(w_): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(R): fn()
    torch.cuda.synchronize()
    return (time.perf_counter()-t0)/R*1e6

def f_ops():
    torch.ops._C.rms_norm(out, x, w, 1e-6)

kfn = native.kernel("rms_norm", "rms_norm_kernel")
def f_native():
    native.launch(kfn, (1,), (32,), (x, w, out, 1, D, 1e-6))

u1 = bench(f_ops)
u2 = bench(f_native)
u3 = bench(lambda: x.mul(1.0))
print(f"torch.ops._C           : {u1:.2f} us")
print(f"ctypes driver launch   : {u2:.2f} us  ({(u1-u2)/u1*100:+.0f}%)")
print(f"torch mul (ref)        : {u3:.2f} us")

# 数值对不对
torch.ops._C.rms_norm(out, x, w, 1e-6); torch.cuda.synchronize()
ref = out.clone()
out.zero_(); f_native(); torch.cuda.synchronize()
print("max diff vs _C:", float((out-ref).abs().max()))
