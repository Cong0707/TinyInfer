import sys, torch, time
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.kernels import native
from tinyinfer.nn import sparse_attn_ref
torch.manual_seed(0)
fn = native.kernel("sparse_attn_sink","sparse_attn_sink")
# 真实形状：H=64 D=512 K=128? 先看模型实际 topk
B,M,H,D,N,K = 1,1,64,512,512,128
q = torch.randn(B,M,H,D, dtype=torch.bfloat16, device="cuda")
kv = torch.randn(B,N,D, dtype=torch.bfloat16, device="cuda")
sink = torch.randn(H, dtype=torch.float32, device="cuda")
idxs = torch.randint(0,N,(B,M,K),dtype=torch.int32,device="cuda")
scale = 1.0/(D**0.5)
nw=B*M*H; blk=min(256,max(32,((nw+31)//32)*32)); wp=blk//32
grid=((nw+wp-1)//wp,)
for _ in range(3): native.launch(fn, grid, (blk,), [q,kv,sink,idxs,torch.empty_like(q),B,M,H,D,N,K,float(scale)])
torch.cuda.synchronize()
t0=time.perf_counter()
for _ in range(20): native.launch(fn, grid, (blk,), [q,kv,sink,idxs,torch.empty_like(q),B,M,H,D,N,K,float(scale)])
torch.cuda.synchronize()
print(f"新 kernel: {(time.perf_counter()-t0)/20*1e6:.0f} us  (H={H} D={D} K={K})")
