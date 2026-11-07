
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.kernels import native
from tinyinfer.nn import sparse_attn_ref
torch.manual_seed(0)
fn = native.kernel("sparse_attn_sink","sparse_attn_sink")
# 只测 H=8, 排除 H=64 的 D=512 干扰
B,M,H,D,N,K = 1,1,8,64,32,8
q = torch.randn(B,M,H,D, dtype=torch.bfloat16, device="cuda")
kv = torch.randn(B,N,D, dtype=torch.bfloat16, device="cuda")
sink = torch.zeros(H, dtype=torch.float32, device="cuda")
idxs = torch.randperm(N)[:K].to(torch.int32).unsqueeze(0).unsqueeze(0).cuda().contiguous()
scale = 1.0/(D**0.5)
ref = sparse_attn_ref(q,kv,sink,idxs,scale)
out = torch.empty_like(q)
nw = B*M*H
blk = min(256, max(32, ((nw+31)//32)*32))
wp = blk//32
grid = (nw + wp - 1)//wp
print(f"nw={nw} blk={blk} warps/block={wp} grid={grid}")
native.launch(fn, (grid,), (blk,), [q,kv,sink,idxs,out,B,M,H,D,N,K,float(scale)])
torch.cuda.synchronize()
print("maxdiff:", (out.float()-ref.float()).abs().max().item())
for h in range(H):
    d = (out[0,0,h].float()-ref[0,0,h].float()).abs().max().item()
    print(f"  head {h}: {d:.3e}")
