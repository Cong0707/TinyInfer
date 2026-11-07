
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.kernels import native
from tinyinfer.nn import sparse_attn_ref
torch.manual_seed(0)
fn = native.kernel("sparse_attn_sink","sparse_attn_sink")
def one(H, D, N, K, fill, rand_sink):
    B,M = 1,1
    q = torch.randn(B,M,H,D, dtype=torch.bfloat16, device="cuda")
    kv = torch.randn(B,N,D, dtype=torch.bfloat16, device="cuda")
    sink = torch.randn(H, dtype=torch.float32, device="cuda") if rand_sink else torch.zeros(H, dtype=torch.float32, device="cuda")
    parts=[torch.randperm(N)[:K-fill].to(torch.int32).tolist()+[-1]*fill]
    idxs = torch.tensor([parts], dtype=torch.int32, device="cuda").contiguous()
    ref = sparse_attn_ref(q,kv,sink,idxs,1.0/(D**0.5))
    out = torch.empty_like(q)
    nw=B*M*H; blk=min(256,max(32,((nw+31)//32)*32)); wp=blk//32
    native.launch(fn, ((nw+wp-1)//wp,), (blk,), [q,kv,sink,idxs,out,B,M,H,D,N,K,float(1.0/(D**0.5))])
    torch.cuda.synchronize()
    d = (out.float()-ref.float()).abs().max().item()
    print(f"H={H:2d} D={D:3d} N={N:3d} K={K:2d} fill={fill:2d} sink={'r' if rand_sink else '0'}: {d:.3e}")
for cfg in ((4,512,128,16,0,True),(8,512,128,16,0,True),(16,512,128,16,2,True),
            (64,512,128,16,0,True),(64,512,128,16,2,True),(64,512,128,16,0,False),
            (64,512,1024,16,0,True)):
    one(*cfg)
