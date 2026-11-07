
import torch, time, sys
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from tinyinfer.kernels import native
from tinyinfer.gemm import fp8_linear, SCALE_FMT, SCALE_DTYPE
from transformers import AutoTokenizer
import kernel as _k

d = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device="cuda", layers=list(range(2)))
w = eng.block(0).ffn.shared.w1
N, K = w.weight.shape
print(f"shape: [{N}, {K}] fp8")
x = torch.randn(1, K, dtype=torch.bfloat16, device="cuda")

y_ref = fp8_linear(x, w.weight, w.scale)            # 现有路径

aq, asc = _k.act_quant(x, 32, SCALE_FMT, SCALE_DTYPE)
A  = aq.view(torch.uint8).reshape(-1)
AS = asc.view(torch.uint8).reshape(-1)
W  = w.weight.view(torch.uint8).reshape(-1)
WS = w.scale.view(torch.uint8).reshape(-1)
y = torch.empty(N, dtype=torch.bfloat16, device="cuda")
fn = native.kernel("fp8_gemv_bs", "fp8_gemv_bs")
native.launch(fn, ((N+15)//16,), (512,), [A, AS, W, WS, y, N, K])
torch.cuda.synchronize()
print("ref:", tuple(y_ref.shape), "ours:", tuple(y.shape))
print("maxdiff:", (y.float() - y_ref.reshape(-1).float()).abs().max().item())

def bench(f, R=300, w_=80):
    for _ in range(w_): f()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(R): f()
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/R*1e6

t_new = bench(lambda: native.launch(fn, ((N+15)//16,), (512,), [A, AS, W, WS, y, N, K]))
t_old = bench(lambda: fp8_linear(x, w.weight, w.scale))
mb = N * K / 1e6
print(f"current fp8_linear : {t_old:7.1f} us  {mb/t_old*1e3:6.0f} GB/s")
print(f"cuda  fp8_gemv_bs  : {t_new:7.1f} us  {mb/t_new*1e3:6.0f} GB/s   {t_old/t_new:.2f}x")
