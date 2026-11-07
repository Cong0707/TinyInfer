
import torch, sys
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
import tinyinfer.deepseek.model as M
from transformers import AutoTokenizer

d = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
# 只加载 2 层：全常驻，无流式，无跨层干扰
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device="cuda", layers=[0, 1])
b = eng.block(0)
print("gpu_layers:", eng.gpu_layers, "max_layers:", eng.max_layers)

torch.manual_seed(0)
hc = b.hc_mult
D = 5120
x = (torch.randn(1, 1, hc, D, dtype=torch.bfloat16, device="cuda") * 0.1)
pre_mix = torch.zeros(1, 1, hc, dtype=torch.float32, device="cuda"); pre_mix[:, :, 0] = 1.0

with torch.inference_mode():
    M._FUSE_HC_PRE = False
    a1, ap1 = b.forward(x, 0, pre_mix)
    torch.cuda.synchronize()
    M._FUSE_HC_PRE = True
    a2, ap2 = b.forward(x, 0, pre_mix)
    torch.cuda.synchronize()
print("x_out   split:", tuple(a1.shape), " fused:", tuple(a2.shape))
print("  maxdiff:", (a1.float()-a2.float()).abs().max().item())
print("pre     maxdiff:", (ap1.float()-ap2.float()).abs().max().item())
