
import ctypes, time, torch
import tinyinfer._C_stable_libtorch  # noqa
from tinyinfer import native

dev = "cuda"
D = 4096
x = torch.randn(1, D, dtype=torch.bfloat16, device=dev)
w = torch.randn(D, dtype=torch.bfloat16, device=dev)
out = torch.empty_like(x)

# 1) torch.ops._C
def f_ops():
    torch.ops._C.rms_norm(out, x, w, 1e-6)

# 2) ctypes driver API（我们 native.py，用空的 sinkhorn 或者 rms_norm cubin）
try:
    kfn = native.kernel("rms_norm", "rms_norm_kernel")
    have_native = True
except Exception as e:
    have_native = False
    print("native rms_norm 不可用:", str(e)[:120])

def f_native():
    # 参数按 kernel 签名打包
    args = (ctypes.c_void_p(x.data_ptr()), ctypes.c_void_p(out.data_ptr()),
            ctypes.c_void_p(w.data_ptr()), ctypes.c_int(1), ctypes.c_int(D),
            ctypes.c_float(1e-6))
    native.launch(kfn, (1,1,1), (32,1,1), args)

def bench(fn, R=3000, w_=300):
    for _ in range(w_): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(R): fn()
    torch.cuda.synchronize()
    return (time.perf_counter()-t0)/R*1e6

u1 = bench(f_ops)
print(f"torch.ops._C           : {u1:.2f} us")
if have_native:
    u2 = bench(f_native)
    print(f"ctypes driver launch   : {u2:.2f} us")
u3 = bench(lambda: x.mul(1.0))
print(f"torch mul (ref)        : {u3:.2f} us")
