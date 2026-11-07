import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.kernels.sparse_attn import sparse_attn_cuda
from tinyinfer.nn import sparse_attn_ref
torch.manual_seed(0)
B,M,H,D,N,K = 1,1,64,512,4096,640
q = torch.randn(B,M,H,D, dtype=torch.bfloat16, device="cuda")
kv = torch.randn(B,N,D, dtype=torch.bfloat16, device="cuda")
sink = torch.randn(H, dtype=torch.float32, device="cuda")*2
idxs = torch.randint(0,N,(B,M,K),dtype=torch.int32,device="cuda")
sc = 1.0/(D**0.5)
ref = sparse_attn_ref(q,kv,sink,idxs,sc)
out = sparse_attn_cuda(q,kv,sink,idxs,sc)
d = (out.float()-ref.float()).abs()
rel = d / ref.float().abs().clamp(min=1e-6)
print(f"maxdiff={d.max().item():.3e}  maxrel={rel.max().item():.3e}")
print(f"ref range: {ref.float().min().item():.3f} ~ {ref.float().max().item():.3f}")
print(f"sink: {[round(v,2) for v in sink[:4].tolist()]}")
# bf16 一个 ulp 在 0.1 量级是多少
import math
v=0.1
print(f"bf16 ulp @0.1: {2**(math.floor(math.log2(v))-8):.2e}")
