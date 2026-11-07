
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.nn import sparse_attn_ref
torch.manual_seed(0)
H,D,N,K = 4,512,128,16
q = torch.randn(1,1,H,D, dtype=torch.bfloat16, device="cuda")
kv = torch.randn(1,N,D, dtype=torch.bfloat16, device="cuda")
sink = torch.randn(H, dtype=torch.float32, device="cuda")
idxs = torch.randperm(N)[:K].to(torch.int32).unsqueeze(0).unsqueeze(0).cuda().contiguous()
scale = 1.0/(D**0.5)
ref = sparse_attn_ref(q,kv,sink,idxs,scale)
print("sink:", [round(v,3) for v in sink.tolist()])
print("ref[0,0,0,:4]:", [round(v,4) for v in ref[0,0,0,:4].tolist()])

# 手工精确复算 head 0
k = kv[0, idxs[0,0].long(), :]      # [K, D]
qf = q[0,0,0].float()
sc = (qf.unsqueeze(0) * k.float()).sum(-1) * scale   # [K]
mx = max(sc.max().item(), sink[0].item())
p = torch.exp(sc - mx)
den = p.sum() + torch.exp(sink[0] - mx)
o = (p.unsqueeze(-1) * k.float()).sum(0) / den
print("手工[0,:4]:", [round(v,4) for v in o[:4].tolist()])
print("ref 与手工 maxdiff:", (ref[0,0,0].float()-o).abs().max().item())
print("sc 范围:", round(sc.min().item(),2), "~", round(sc.max().item(),2), " sink:", round(sink[0].item(),2))
