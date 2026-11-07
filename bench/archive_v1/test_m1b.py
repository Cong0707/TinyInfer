import sys, time, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\reference\official")
from tinyinfer.weights import ShardStore
from tinyinfer.moe_fused import fused_fp4_expert_ffn
from tinyinfer.quant import dequant_fp4_block

s = ShardStore(r"E:\Develop\Deepseek\tinyinfer\cache\weights")
w1 = s.get("layers.0.ffn.experts.0.w1.weight").cuda(); s1 = s.get("layers.0.ffn.experts.0.w1.scale").cuda()
w2 = s.get("layers.0.ffn.experts.0.w2.weight").cuda(); s2 = s.get("layers.0.ffn.experts.0.w2.scale").cuda()
w3 = s.get("layers.0.ffn.experts.0.w3.weight").cuda(); s3 = s.get("layers.0.ffn.experts.0.w3.scale").cuda()
x = torch.randn(1, 5120, device="cuda", dtype=torch.bfloat16)
out = fused_fp4_expert_ffn(x, w1, s1, w3, s3, w2, s2, torch.ones(1, device="cuda"))
# 参照：dequant + torch
g = x.to(torch.bfloat16) @ dequant_fp4_block(w1, s1).to(torch.bfloat16).T
u = x.to(torch.bfloat16) @ dequant_fp4_block(w3, s3).to(torch.bfloat16).T
y = torch.nn.functional.silu(g.clamp(max=10)) * u.clamp(-10, 10)
ref = y.to(torch.bfloat16) @ dequant_fp4_block(w2, s2).to(torch.bfloat16).T
torch.cuda.synchronize()
rel = ((out - ref).abs().max() / ref.abs().max()).item()
t0 = time.time()
for _ in range(20):
    fused_fp4_expert_ffn(x, w1, s1, w3, s3, w2, s2, torch.ones(1, device="cuda"))
torch.cuda.synchronize()
tm = (time.time() - t0) / 20
print(f"fused M=1: {tm*1000:.0f}us  rel {rel:.2e}")
