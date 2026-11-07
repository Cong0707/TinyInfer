import sys, time, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\reference\official")
from tinyinfer.weights import ShardStore
from tinyinfer.moe_fused import fused_fp4_expert_ffn
import kernel as k

s = ShardStore(r"E:\Develop\Deepseek\tinyinfer\cache\weights")
w1 = s.get("layers.0.ffn.experts.0.w1.weight").cuda(); s1 = s.get("layers.0.ffn.experts.0.w1.scale").cuda()
w2 = s.get("layers.0.ffn.experts.0.w2.weight").cuda(); s2 = s.get("layers.0.ffn.experts.0.w2.scale").cuda()
w3 = s.get("layers.0.ffn.experts.0.w3.weight").cuda(); s3 = s.get("layers.0.ffn.experts.0.w3.scale").cuda()
x = torch.randn(1, 5120, device="cuda", dtype=torch.bfloat16)
for _ in range(3):
    aq, asc = k.act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu)
    g = k.fp4_gemm(aq, asc, w1, s1, torch.float8_e8m0fnu, act_block_size=32)
    u = k.fp4_gemm(aq, asc, w3, s3, torch.float8_e8m0fnu, act_block_size=32)
    y = (torch.nn.functional.silu(g.clamp(max=10)) * u.clamp(-10, 10))
    hq, hs = k.act_quant(y.to(torch.bfloat16), 32, "ue8m0", torch.float8_e8m0fnu)
    ref = k.fp4_gemm(hq, hs, w2, s2, torch.float8_e8m0fnu, act_block_size=32)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(20):
    out = fused_fp4_expert_ffn(x, w1, s1, w3, s3, w2, s2, torch.ones(1, device="cuda"))
torch.cuda.synchronize()
tm = (time.time() - t0) / 20
t0 = time.time()
for _ in range(20):
    aq, asc = k.act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu)
    g = k.fp4_gemm(aq, asc, w1, s1, torch.float8_e8m0fnu, act_block_size=32)
    u = k.fp4_gemm(aq, asc, w3, s3, torch.float8_e8m0fnu, act_block_size=32)
    y = (torch.nn.functional.silu(g.clamp(max=10)) * u.clamp(-10, 10))
    hq, hs = k.act_quant(y.to(torch.bfloat16), 32, "ue8m0", torch.float8_e8m0fnu)
    ref = k.fp4_gemm(hq, hs, w2, s2, torch.float8_e8m0fnu, act_block_size=32)
torch.cuda.synchronize()
to = (time.time() - t0) / 20
rel = ((out - ref).abs().max() / ref.abs().max()).item()
print(f"fused M=1: {tm*1000:.0f}us  official: {to*1000:.0f}us  speedup {to/tm:.2f}x  rel {rel:.2e}")
