
import torch, sys, traceback
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
import tinyinfer.gemm as G

d = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device="cuda", layers=list(range(2)))
shared = eng.block(0).ffn.shared
w = shared.w1
print("shared.w1 type:", type(w).__name__)
if hasattr(w, "weight"):
    print("  weight:", tuple(w.weight.shape), w.weight.dtype)
    print("  scale :", tuple(w.scale.shape), w.scale.dtype)
    x = torch.randn(1, w.weight.shape[1], dtype=torch.bfloat16, device="cuda")
    from tinyinfer.gemv_fp8 import gemv_fp8
    try:
        y = gemv_fp8(x, w.weight, w.scale)
        print("  gemv_fp8 ->", tuple(y.shape), y.dtype)
    except Exception:
        print("  gemv_fp8 FAILED:")
        traceback.print_exc()
