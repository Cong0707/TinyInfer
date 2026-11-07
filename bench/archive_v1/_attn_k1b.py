
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.kernels import native
from tinyinfer.nn import sparse_attn_ref
torch.manual_seed(0)
B,M,H,D,N,K = 1,1,1,64,32,1
q = torch.randn(B,M,H,D, dtype=torch.bfloat16, device="cuda")
kv = torch.randn(B,N,D, dtype=torch.bfloat16, device="cuda")
sink = torch.zeros(H, dtype=torch.float32, device="cuda")
idxs = torch.tensor([[[7]]], dtype=torch.int32, device="cuda")
scale = 1.0/(D**0.5)
ref = sparse_attn_ref(q,kv,sink,idxs,scale)
out = torch.empty_like(q)
fn = native.kernel("sparse_attn_sink","sparse_attn_sink")
sm = (3+32+2)*4
native.launch(fn,(B*M,H),(128,),[q,kv,sink,idxs,out,B,M,H,D,N,K,float(scale)],shared=sm)
torch.cuda.synchronize()
print("ref[:6]:", [round(v,4) for v in ref[0,0,0,:6].tolist()])
print("out[:6]:", [round(v,4) for v in out[0,0,0,:6].tolist()])
print("ratio:", [round((out[0,0,0,i]/ref[0,0,0,i]).item(),4) if ref[0,0,0,i].abs()>1e-6 else 'na' for i in range(6)])
print("kv[7][:6]:", [round(v,4) for v in kv[0,7,:6].tolist()])
print("q[:6]:", [round(v,4) for v in q[0,0,0,:6].tolist()])
# 手工算 sink=0 单 slot
sc = (q[0,0,0].float() @ kv[0,7].float()) * scale
import math
mx = max(sc.item(), 0.0)
p = math.exp(sc.item()-mx); den = p + math.exp(0.0-mx)
manual = kv[0,7].float() * p / den
print("手工[:6]:", [round(v,4) for v in manual[:6].tolist()])
