
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.kernels import native
from tinyinfer.nn import sparse_attn_ref

torch.manual_seed(0)
B, M, H, D, N, K = 1, 1, 1, 64, 32, 4
q = torch.randn(B, M, H, D, dtype=torch.bfloat16, device="cuda")
kv = torch.randn(B, N, D, dtype=torch.bfloat16, device="cuda")
sink = torch.zeros(H, dtype=torch.float32, device="cuda")   # sink=0，简化
idxs = torch.tensor([[[0,1,2,3]]], dtype=torch.int32, device="cuda")
scale = 1.0 / (D ** 0.5)

ref = sparse_attn_ref(q, kv, sink, idxs, scale)
out = torch.empty_like(q)
fn = native.kernel("sparse_attn_sink", "sparse_attn_sink")
blk = 128
smem = 3 * 4 + 32 * 4
native.launch(fn, (B * M, H), (blk,), [q, kv, sink, idxs, out, B, M, H, D, N, K, float(scale)], shared=smem)
torch.cuda.synchronize()
print("ref[:8]:", [round(v,4) for v in ref[0,0,0,:8].tolist()])
print("out[:8]:", [round(v,4) for v in out[0,0,0,:8].tolist()])
print("maxdiff:", (out.float()-ref.float()).abs().max().item())

# 纯 python 手工算，确认参考本身
k4 = kv[0, idxs[0,0].long()]           # [4, D]
sc = (q[0,0,0].float() @ k4.float().T) * scale
mx = max(sc.max().item(), 0.0)
p = torch.exp(sc - mx)
den = p.sum() + torch.exp(torch.tensor(0.0) - mx)
o = (p.unsqueeze(-1) * k4.float()).sum(0) / den
print("手工[:8]:", [round(v,4) for v in o[:8].tolist()])
