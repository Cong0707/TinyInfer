
import torch
from tinyinfer.kernels import native
from tinyinfer.deepseek.hc import _hc_post_ref
from tinyinfer.kernels.tilelang.mhc.post_kernel import mhc_post_fwd

torch.manual_seed(0)
S, T, hc, H = 1, 4, 4, 256
x    = torch.randn(S, T, H, dtype=torch.bfloat16, device="cuda")
res  = torch.randn(S, T, hc, H, dtype=torch.bfloat16, device="cuda")
post = torch.randn(S, T, hc, dtype=torch.float32, device="cuda")
comb = torch.randn(S, T, hc, hc, dtype=torch.float32, device="cuda")

# 官方
off = mhc_post_fwd(x, res, post.unsqueeze(-1), comb)
torch.cuda.synchronize()
print("official:", tuple(off.shape), off.dtype)

# 我们的 kernel（压平成 M 维）
M = S * T
xc = x.reshape(M, H).contiguous()
rc = res.reshape(M, hc, H).contiguous()
pc = post.reshape(M, hc).contiguous()
cc = comb.reshape(M, hc, hc).contiguous()
out = torch.empty((M, hc, H), dtype=torch.bfloat16, device="cuda")
n = M * hc * H
native.launch(native.kernel("hc", "hc_post_bf"), ((n+255)//256,), (256,),
              [xc, rc, pc, cc, out, M, hc, H])
torch.cuda.synchronize()
ours = out.view(S, T, hc, H)

ref = _hc_post_ref(x, res, post, comb)
print("torch ref vs official maxdiff:", float((ref.float()-off.float()).abs().max()))
print("cubin     vs official maxdiff:", float((ours.float()-off.float()).abs().max()))
print("cubin scale:", float(off.float().abs().max()))
