
import torch, sys, time
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\reference\official")
from tinyinfer.kernels import native
import kernel as _k
from tinyinfer.gemm import SCALE_FMT, SCALE_DTYPE, fp8_linear

def bench(f, R=200, w_=50):
    for _ in range(w_): f()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(R): f()
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/R*1e6

for (N, K) in ((32768, 1280), (5120, 2304), (2304, 5120)):
    w = torch.randn(N, K, dtype=torch.float16, device="cuda").to(torch.float8_e4m3fn)
    s = torch.ones(N//32, K//32, dtype=torch.float8_e8m0fnu, device="cuda")
    x = torch.randn(1, K, dtype=torch.bfloat16, device="cuda")
    ref = fp8_linear(x, w, s)
    aq, asc = _k.act_quant(x, 32, SCALE_FMT, SCALE_DTYPE)
    y = torch.empty(N, dtype=torch.bfloat16, device="cuda")
    fn = native.kernel("fp8_gemv_bs","fp8_gemv_bs")
    args = [aq.view(torch.uint8).reshape(-1), asc.view(torch.uint8).reshape(-1),
            w.view(torch.uint8).reshape(-1), s.view(torch.uint8).reshape(-1), y, N, K]
    native.launch(fn, ((N+15)//16,), (512,), args)
    torch.cuda.synchronize()
    d = (y.float()-ref.reshape(-1).float()).abs().max().item()
    t = bench(lambda: native.launch(fn, ((N+15)//16,), (512,), args))
    mb = N*K/1e6
    print(f"[{N:6d},{K:5d}] {t:7.1f}us {mb/t*1e3:5.0f} GB/s  maxdiff={d:.2e}")
