import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel

eng = Engine(r"E:\Develop\Deepseek\tinyinfer\cache\weights", tier=OptLevel.BALANCED,
             device="cuda", layers=[7])
print("engine ok")
b = eng.block(7)
print("block7 built:", b.attn.__class__.__name__, "quant:", b.quant)
x = torch.randn(1, 1, 4, 5120, dtype=torch.bfloat16, device="cuda") * 0.02
pre = torch.zeros(1, 1, 4, dtype=torch.float32, device="cuda"); pre[:, :, 0] = 1.0
with torch.inference_mode():
    h, _ = b.forward(x, 0, pre)
print("forward ok", h.shape)
