import sys
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
import torch
import tinyinfer.torch_compat  # noqa: F401
from tinyinfer import gemv_fp4, gemv_fp4_native, native

def bench(fn, n=50, warm=10):
    with torch.inference_mode():
        for _ in range(warm): fn()
        torch.cuda.synchronize()
        e0 = torch.cuda.Event(True); e1 = torch.cuda.Event(True)
        e0.record()
        for _ in range(n): fn()
        e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1)/n

N, K = 4096, 5120
w = torch.randint(0, 256, (N, K//2), dtype=torch.uint8, device="cuda")
# e8m0 scale bytes: 127 = 2^0
ws_bytes = torch.full((N, K//32), 127, dtype=torch.uint8, device="cuda")
x = torch.randn(1, K, dtype=torch.bfloat16, device="cuda") * 0.1

# tilelang reference
y_ref = gemv_fp4.gemv_fp4(x, w, ws_bytes.view(torch.uint8).view(-1, K//32)[:N] if False else ws_bytes)
torch.cuda.synchronize()
print("tilelang out[0:3]:", y_ref[0][:3].tolist())

# need the raw activation path: bypass act_quant by calling the native kernel directly
a = x.reshape(-1).float()
asc = torch.ones(K//32, dtype=torch.float32, device="cuda")
out = torch.empty(N, dtype=torch.bfloat16, device="cuda")
native.launch(native.kernel("gemv_fp4_e8m0","gemv_fp4_fast"), ((N+7)//8,), (256,),
              [a, asc, w, ws_bytes, out, N, K])
torch.cuda.synchronize()
print("native   out[0:3]:", out[:3].tolist())
print("diff amax:", float((y_ref[0].float()-out.float()).abs().max()))
print()
ms_t = bench(lambda: gemv_fp4.gemv_fp4(x, w, ws_bytes))
ms_n = bench(lambda: native.launch(native.kernel("gemv_fp4_e8m0","gemv_fp4_fast"),
                                   ((N+7)//8,), (256,), [a, asc, w, ws_bytes, out, N, K]))
by = N*K*0.5
print(f"tilelang: {ms_t*1000:8.1f} us -> {by/(ms_t/1000)/1e9:6.0f} GB/s")
print(f"native  : {ms_n*1000:8.1f} us -> {by/(ms_n/1000)/1e9:6.0f} GB/s  x{ms_t/ms_n:.2f}")
