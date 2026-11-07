
import torch, sys, time
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer

d = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device="cuda", layers=list(range(2)))
w = eng.block(0).ffn.shared.w1
x = torch.randn(1, w.weight.shape[1], dtype=torch.bfloat16, device="cuda")

from tinyinfer.gemv_fp8 import fused_fp8_gemv
from tinyinfer.official_kernel_ref import act_quant, fp8_gemm
import kernel as _k
from tinyinfer.gemm import SCALE_FMT, SCALE_DTYPE

def bench(fn, R=200, w_=50):
    for _ in range(w_): fn()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(R): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/R*1e6

y1 = fused_fp8_gemv(x, w.weight, w.scale)
aq, asc = _k.act_quant(x, 32, SCALE_FMT, SCALE_DTYPE)
y2 = _k.fp8_gemm(aq, asc, w.weight, w.scale, SCALE_DTYPE)
torch.cuda.synchronize()
print("shape:", tuple(y1.shape), "maxdiff:", (y1.float()-y2.float()).abs().max().item())
t1 = bench(lambda: fused_fp8_gemv(x, w.weight, w.scale))
t2 = bench(lambda: _k.fp8_gemm(aq, asc, w.weight, w.scale, SCALE_DTYPE))
gb = w.weight.numel()/1e9
print(f"gemv_fp8 : {t1:7.1f} us  {gb/t1*1e6:.0f} GB/s")
print(f"fp8_gemm : {t2:7.1f} us  {gb/t2*1e6:.0f} GB/s   {t2/t1:.2f}x")
