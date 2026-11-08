"""M=1 兜底失败的具体原因。"""
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.gemm import fp8_linear

# 构造 wq_a 的形状
N, K = 32768, 1280
w = torch.randn(N, K, device="cuda").to(torch.float8_e4m3fn)
ws = torch.randn((N + 127) // 128, (K + 31) // 32, device="cuda").to(torch.float8_e4m3fn)
x = torch.randn(1, K, device="cuda", dtype=torch.bfloat16)
print("w=%s ws=%s x=%s" % (tuple(w.shape), tuple(ws.shape), tuple(x.shape)), flush=True)
try:
    y = fp8_linear(x, w, ws, 32)
    print("OK", tuple(y.shape), flush=True)
except Exception as ex:
    import traceback; traceback.print_exc()
