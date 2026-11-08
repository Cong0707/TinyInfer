import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
import tinyinfer.nn as NN

# 直接对比 apply_rope：单行 freqs[1,32] vs 双行 freqs[2,1,32]，x 相同
torch.manual_seed(0)
x1 = torch.randn(1, 1, 64, 64, device="cuda", dtype=torch.bfloat16) * 0.5
x2 = torch.cat([x1, x1], 0).clone()
f1 = torch.randn(1, 32, device="cuda", dtype=torch.complex64)
f2 = f1.view(1, 1, 32).expand(2, 1, 32).contiguous()

y1 = NN.apply_rope(x1.clone(), f1, inverse=False)
y2 = NN.apply_rope(x2.clone(), f2, inverse=False)
print("rope row0 vs single = %.6f" % float((y2[0:1].float() - y1.float()).abs().max()), flush=True)
print("rope rows identical = %.6f" % float((y2[0].float() - y2[1].float()).abs().max()), flush=True)
