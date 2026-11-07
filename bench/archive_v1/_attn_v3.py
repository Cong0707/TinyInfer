
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.kernels import native
from tinyinfer.nn import sparse_attn_ref
torch.manual_seed(0)
fn = native.kernel("sparse_attn_sink","sparse_attn_sink")

def launch(q, kv, sink, idxs, out):
    B,M,H,D = q.shape; N = kv.shape[1]; K = idxs.shape[-1]
    nw = B*M*H
    blk = min(256, max(32, ((nw+31)//32)*32))
    wp = blk//32
    grid = (nw + wp - 1)//wp
    native.launch(fn, (grid,), (blk,), [q,kv,sink,idxs,out,B,M,H,D,N,K,float(1.0/(D**0.5))])

def run(tag, B,M,H,D,N,K, rand_sink, fill):
    q = torch.randn(B,M,H,D, dtype=torch.bfloat16, device="cuda")
    kv = torch.randn(B,N,D, dtype=torch.bfloat16, device="cuda")
    sink = torch.randn(H, dtype=torch.float32, device="cuda") if rand_sink else torch.zeros(H, dtype=torch.float32, device="cuda")
    parts=[[t for t in torch.randperm(N)[:K-fill].tolist()] + [-1]*fill for _ in range(M)]
    idxs = torch.tensor([parts], dtype=torch.int32, device="cuda").contiguous()
    ref = sparse_attn_ref(q,kv,sink,idxs,1.0/(D**0.5))
    out = torch.empty_like(q)
    launch(q,kv,sink,idxs,out); torch.cuda.synchronize()
    print(f"{tag:32s} maxdiff={(out.float()-ref.float()).abs().max().item():.3e}")

run("K=1",              1,1,1,64,32,1,  False,0)
run("K=3",              1,1,1,64,32,3,  False,0)
run("K=16 随机sink",     1,1,1,64,32,16, True,0)
run("K=16 填充3",        1,1,1,64,32,16, True,3)
run("K=16 全填充",       1,1,1,64,32,16, True,16)
run("H=8",              1,1,8,128,64,16, True,2)
run("H=64 D=512 真实",   1,1,64,512,128,16, True,2)
run("M=4 H=8",          1,4,8,128,64,16, True,3)
run("B=2 M=2 H=4",      2,2,4,64,32,8,  True,0)
run("H=64 D=512 长K=64", 1,1,64,512,128,64, True,5)
