import sys, torch, time
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.kernels.sparse_attn import sparse_attn_cuda
from tinyinfer.nn import sparse_attn_ref
torch.manual_seed(0)
def run(B,M,H,D,N,K,sink_r,fill):
    q = torch.randn(B,M,H,D, dtype=torch.bfloat16, device="cuda")
    kv = torch.randn(B,N,D, dtype=torch.bfloat16, device="cuda")
    sink = torch.randn(H, dtype=torch.float32, device="cuda") if sink_r else torch.zeros(H, dtype=torch.float32, device="cuda")
    parts=[[t for t in torch.randperm(N)[:K-fill].tolist()]+[-1]*fill for _ in range(M)]
    idxs = torch.tensor([parts], dtype=torch.int32, device="cuda")
    sc = 1.0/(D**0.5)
    ref = sparse_attn_ref(q,kv,sink,idxs,sc)
    out = sparse_attn_cuda(q,kv,sink,idxs,sc)
    if out is None:
        print(f"H{H} D{D} N{N} K{K}: rejected"); return
    d=(out.float()-ref.float()).abs().max().item()
    print(f"H{H} D{D} N{N} K{K} fill{fill} sink{'r' if sink_r else '0'}: maxdiff={d:.3e}")
run(1,1,64,512,4096,640,True,0)
run(1,1,64,512,4096,640,True,64)
run(1,1,1,64,32,1,False,0)
run(1,1,8,128,64,16,True,3)
run(1,4,8,128,64,16,True,3)
run(1,1,64,512,4096,640,False,0)
