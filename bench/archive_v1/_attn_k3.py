
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.kernels import native
from tinyinfer.nn import sparse_attn_ref
torch.manual_seed(0)
B,M,H,D,N = 1,1,1,64,32
q = torch.randn(B,M,H,D, dtype=torch.bfloat16, device="cuda")
kv = torch.randn(B,N,D, dtype=torch.bfloat16, device="cuda")
sink = torch.zeros(H, dtype=torch.float32, device="cuda")
scale = 1.0/(D**0.5)
fn = native.kernel("sparse_attn_sink","sparse_attn_sink")
sm = (3+32+2)*4
perm = torch.randperm(N)
print("perm dtype:", perm.dtype)
for K in (1,2,3,4,5,8,16):
    # 关键：独立构造 + contiguous
    idxs = perm[:K].clone().contiguous().unsqueeze(0).unsqueeze(0).int().cuda().contiguous()
    ref = sparse_attn_ref(q,kv,sink,idxs,scale)
    out = torch.empty_like(q)
    native.launch(fn,(B*M,H),(128,),[q,kv,sink,idxs,out,B,M,H,D,N,K,float(scale)],shared=sm)
    torch.cuda.synchronize()
    print(f"K={K:2d} idx={idxs.flatten().tolist()[:6]}: maxdiff={(out.float()-ref.float()).abs().max().item():.3e}")
