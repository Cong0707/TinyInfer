
import torch, sys, subprocess
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
from tinyinfer.gemv_fp8 import fused_fp8_gemv

d = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device="cuda", layers=list(range(2)))
w = eng.block(0).ffn.shared.w1
x = torch.randn(1, w.weight.shape[1], dtype=torch.bfloat16, device="cuda")
print("shape:", tuple(w.weight.shape), "rows test")
for rows in (16, 8, 4, 2):
    try:
        y = fused_fp8_gemv(x, w.weight, w.scale, rows=rows)
        torch.cuda.synchronize()
        print(f"  rows={rows:2d}: OK  {tuple(y.shape)}  finite={bool(torch.isfinite(y).all())}")
    except Exception as e:
        print(f"  rows={rows:2d}: FAIL {type(e).__name__} {str(e)[:80]}")
        break
