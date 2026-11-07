
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.kernels import native
from tinyinfer.nn import sparse_attn_ref
torch.manual_seed(0)
fn = native.kernel("sparse_attn_sink","sparse_attn_sink")
def one(D, K=8):
    B,M,H,N = 1,1,8,32
    q = torch.randn(B,M,H,D, dtype=torch.bfloat16, device="cuda")
    kv = torch.randn(B,N,D, dtype=torch.bfloat16, device="cuda")
    sink = torch.zeros(H, dtype=torch.float32, device="cuda")
    idxs = torch.randperm(N)[:K].to(torch.int32).unsqueeze(0).unsqueeze(0).cuda().contiguous()
    ref = sparse_attn_ref(q,kv,sink,idxs,1.0/(D**0.5))
    out = torch.empty_like(q)
    nw=B*M*H; blk=min(256,max(32,((nw+31)//32)*32)); wp=blk//32
    native.launch(fn, ((nw+wp-1)//wp,), (blk,), [q,kv,sink,idxs,out,B,M,H,D,N,K,float(1.0/(D**0.5))])
    torch.cuda.synchronize()
    print(f"D={D:4d} (nlocal={(D+31)//32:2d}): maxdiff={(out.float()-ref.float()).abs().max().item():.3e}")
for D in (64, 128, 256, 384, 512):
    one(D)
