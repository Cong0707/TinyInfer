import sys, torch, time
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.kernels.sparse_attn import sparse_attn_cuda
torch.manual_seed(0)
def bench(H,D,N,K,R=30):
    B,M=1,1
    q = torch.randn(B,M,H,D, dtype=torch.bfloat16, device="cuda")
    kv = torch.randn(B,N,D, dtype=torch.bfloat16, device="cuda")
    sink = torch.randn(H, dtype=torch.float32, device="cuda")
    idxs = torch.randint(0,N,(B,M,K),dtype=torch.int32,device="cuda")
    sc=1.0/(D**0.5)
    for _ in range(5): sparse_attn_cuda(q,kv,sink,idxs,sc)
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(R): sparse_attn_cuda(q,kv,sink,idxs,sc)
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/R*1e6
print(f"真实 K=640: {bench(64,512,4096,640):.0f} us")
print(f"窗口 K=128: {bench(64,512,4096,128):.0f} us")
