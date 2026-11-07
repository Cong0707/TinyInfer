
import torch
from tinyinfer.kernels import native

def ref_pair(x, pre_mix, w, eps):
    h = torch.sum(pre_mix.unsqueeze(-1) * x.float(), dim=1).to(x.dtype)   # 消 hc 维
    xf = h.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w.float()
    return y.to(x.dtype)

torch.manual_seed(0)
for (M, hc, D) in [(1,4,64), (1,4,3072), (1,4,7168), (8,4,1024)]:
    x    = torch.randn(M, hc, D, dtype=torch.bfloat16, device="cuda")
    pre  = torch.randn(M, hc, dtype=torch.float32, device="cuda")
    w    = torch.randn(D, dtype=torch.bfloat16, device="cuda")
    eps  = 1e-6
    ref = ref_pair(x, pre, w, eps)
    out = torch.empty((M, D), dtype=torch.bfloat16, device="cuda")
    native.launch(native.kernel("hc_pre_rms_norm", "hc_pre_rms_norm_bf"),
                  (M,), (256,), [x, pre, w, out, M, hc, D, float(eps)], shared=D*4)
    torch.cuda.synchronize()
    d = (out.float()-ref.float()).abs().max().item()
    print(f"M={M} D={D:<5} maxdiff={d:.3e} bit={bool((out==ref).all())}")
