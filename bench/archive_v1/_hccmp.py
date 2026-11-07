
import torch
from tinyinfer.kernels import native

torch.manual_seed(0)
M, hc, D = 4, 4, 128
x    = torch.randn(M, D, dtype=torch.bfloat16, device="cuda")
res  = torch.randn(M, hc, D, dtype=torch.bfloat16, device="cuda")
post = torch.randn(M, hc, dtype=torch.float32, device="cuda")
comb = torch.randn(M, hc, hc, dtype=torch.float32, device="cuda")

# 参考：hc.py 的 torch 表达式原样
ref = (post.unsqueeze(-1) * x.unsqueeze(-2)
       + torch.sum(comb.unsqueeze(-1) * res.unsqueeze(-2).float(), dim=2)).to(x.dtype)

out = torch.empty((M, hc, D), dtype=x.dtype, device="cuda")
n = M * hc * D
native.launch(native.kernel("hc", "hc_post_bf"), ((n + 255)//256,), (256,),
              [x, res, post.contiguous(), comb.contiguous(), out, M, hc, D])
torch.cuda.synchronize()
print("shape:", tuple(out.shape), "ref", tuple(ref.shape))
print("bit-identical:", bool((out == ref).all()))
print(f"max abs diff : {(out.float()-ref.float()).abs().max().item():.3e}")
print(f"ref max      : {ref.float().abs().max().item():.3e}")
