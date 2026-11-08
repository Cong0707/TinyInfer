import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.kernels.quant import act_quant

torch.manual_seed(0)
o1 = torch.randn(1, 1, 4, 512, device="cuda", dtype=torch.bfloat16) * 0.5
o2 = torch.cat([o1, o1], 0).clone()
a1 = o1.clone(); act_quant(a1, 32, "ue8m0", torch.float8_e8m0fnu, True)
a2 = o2.clone(); act_quant(a2, 32, "ue8m0", torch.float8_e8m0fnu, True)
print("act_quant row0 vs single = %.6f" % float((a2[0:1].float() - a1.float()).abs().max()), flush=True)
print("act_quant rows identical = %.6f" % float((a2[0].float() - a2[1].float()).abs().max()), flush=True)
