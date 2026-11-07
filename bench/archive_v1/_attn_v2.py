
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.kernels import native
from tinyinfer.nn import sparse_attn_ref
torch.manual_seed(0)
fn = native.kernel("sparse_attn_sink","sparse_attn_sink")

def run(tag, B,M,H,D,N,K, sink_mode, fill):
    q = torch.randn(B,M,H,D, dtype=torch.bfloat16, device="cuda")
    kv = torch.randn(B,N,D, dtype=torch.bfloat16, device="cuda")
    sink = torch.randn(H, dtype=torch.float32, device="cuda") if sink_mode else torch.zeros(H, dtype=torch.float32, device="cuda")
    nv = K - fill
    parts=[]
    for _ in range(M):
        t = torch.randperm(N)[:nv].to(torch.int32).tolist() + [-1]*fill
        parts.append(t)
    idxs = torch.tensor([parts], dtype=torch.int32, device="cuda").contiguous()
    scale = 1.0/(D**0.5)
    ref = sparse_attn_ref(q,kv,sink,idxs,scale)
    out = torch.empty_like(q)
    nw = B*M*H
    blk = min(256, ((nw+31)//32)*32) or 32
    native.launch(fn, ((nw+blk//32-1)//(blk//32),), (blk,), [q,kv,sink,idxs,out,B,M,H,D,N,K,float(scale)])
    torch.cuda.synchronize()
    print(f"{tag:34s} maxdiff={(out.float()-ref.float()).abs().max().item():.3e}")

run("K=1 sink=0",       1,1,1,64,32,1,  False, 0)
run("K=2 sink=0",       1,1,1,64,32,2,  False, 0)
run("K=3 sink=0",       1,1,1,64,32,3,  False, 0)
run("K=16 sink随机",     1,1,1,64,32,16, True,  0)
run("K=16 填充3 sink随机", 1,1,1,64,32,16, True,  3)
run("K=16 全填充",       1,1,1,64,32,16, True, 16)
run("H=64 D=512 (真实)", 1,1,64,512,128,16, True, 2)
run("M=4 H=8 D=128",    1,4,8,128,64,16, True, 3)
run("B=2",              2,1,4,64,32,8,  True,  0)
