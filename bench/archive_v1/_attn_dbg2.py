
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.kernels import native
from tinyinfer.nn import sparse_attn_ref

torch.manual_seed(0)
B,M,H,D,N,K = 1,1,1,64,32,16
q = torch.randn(B,M,H,D, dtype=torch.bfloat16, device="cuda")
kv = torch.randn(B,N,D, dtype=torch.bfloat16, device="cuda")
sink = torch.randn(H, dtype=torch.float32, device="cuda")
print("sink =", sink.tolist())

for nvalid in (16, 13, 0):
    idxs = torch.randperm(N)[:K].unsqueeze(0).unsqueeze(0).int().cuda()
    n_fill = K - nvalid
    if n_fill: idxs[0,0,-n_fill:] = -1
    scale = 1.0/(D**0.5)
    ref = sparse_attn_ref(q, kv, sink, idxs, scale)
    out = torch.empty_like(q)
    fn = native.kernel("sparse_attn_sink","sparse_attn_sink")
    native.launch(fn,(B*M,H),(128,),[q,kv,sink,idxs,out,B,M,H,D,N,K,float(scale)],shared=3*4+32*4)
    torch.cuda.synchronize()
    d = (out.float()-ref.float()).abs().max().item()
    print(f"有效 slot={nvalid:2d} (填充{n_fill:2d}): maxdiff={d:.3e}  ref[0]={ref[0,0,0,0].item():.4f} out[0]={out[0,0,0,0].item():.4f}")
