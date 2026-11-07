import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from tinyinfer.gemv_fp8 import fused_fp8_gemv

torch.manual_seed(0)
eng = Engine(r"E:\Develop\Deepseek\tinyinfer\cache\weights", tier=OptLevel.BALANCED,
             device="cuda", layers=[0])
a = eng.block(0).attn
x = torch.randn(1, 5120, dtype=torch.bfloat16, device="cuda") * 0.02
with torch.inference_mode():
    w = a.wq_a
    print("w type:", type(w).__name__, "weight:", None if not hasattr(w, "weight") else w.weight.shape,
          "scale:", None if not hasattr(w, "scale") else (w.scale.shape, w.scale.dtype))
    y = fused_fp8_gemv(x, w.weight, w.scale)
    print("gemv out:", y.shape, y.dtype, float(y.abs().mean()))
    ref = w.forward(x)
    print("ref out:", ref.shape, float(ref.abs().mean()))
    print("max diff:", float((y - ref).abs().max()))
