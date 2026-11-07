
import torch
from tinyinfer.kernels import native
from tinyinfer.deepseek.hc import _hc_post_ref

torch.manual_seed(0)
for (M, hc, D) in [(4,4,128), (1,4,3072), (8,4,512), (1,4,64)]:
    x    = torch.randn(M, D, dtype=torch.bfloat16, device="cuda")
    res  = torch.randn(M, hc, D, dtype=torch.bfloat16, device="cuda")
    post = torch.randn(M, hc, dtype=torch.float32, device="cuda")
    comb = torch.randn(M, hc, hc, dtype=torch.float32, device="cuda")

    ref = _hc_post_ref(x, res, post, comb)
    out = torch.empty((M, hc, D), dtype=x.dtype, device="cuda")
    n = M * hc * D
    native.launch(native.kernel("hc", "hc_post_bf"), ((n + 255)//256,), (256,),
                  [x, res, post.contiguous(), comb.contiguous(), out, M, hc, D])
    torch.cuda.synchronize()
    d = (out.float() - ref.float()).abs().max().item()
    print(f"M={M:<2} hc={hc} D={D:<5} shape={tuple(out.shape)}  maxdiff={d:.3e}  bit={bool((out==ref).all())}")
