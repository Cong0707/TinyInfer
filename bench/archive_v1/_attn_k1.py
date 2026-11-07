
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.kernels import native
from tinyinfer.nn import sparse_attn_ref
torch.manual_seed(0)
# 单 slot，K=1，看是不是纯粹的单点计算问题
B,M,H,D,N,K = 1,1,1,64,32,1
q = torch.randn(B,M,H,D, dtype=torch.bfloat16, device="cuda")
kv = torch.randn(B,N,D, dtype=torch.bfloat16, device="cuda")
sink = torch.zeros(H, dtype=torch.float32, device="cuda")
idxs = torch.tensor([[[5]]], dtype=torch.int32, device="cuda")
scale = 1.0/(D**0.5)
ref = sparse_attn_ref(q,kv,sink,idxs,scale)
out = torch.empty_like(q)
fn = native.kernel("sparse_attn_sink","sparse_attn_sink")
native.launch(fn,(B*M,H),(128,),[q,kv,sink,idxs,out,B,M,H,D,N,K,float(scale)],shared=3*4+32*4)
torch.cuda.synchronize()
print("K=1 maxdiff:", (out.float()-ref.float()).abs().max().item())
print("ref[0][:4]:", [round(v,4) for v in ref[0,0,0,:4].tolist()])
print("out[0][:4]:", [round(v,4) for v in out[0,0,0,:4].tolist()])
print("kv[5][:4]:", [round(v,4) for v in kv[0,5,:4].tolist()])
