
import torch
from tinyinfer.kernels import native
from tinyinfer.nn import rms_norm, apply_rope

torch.manual_seed(0)
def ref(x, w, freqs, rot, eps, S):
    y = rms_norm(x, w, eps)
    v = y.view(1, S, -1) if S > 0 else y
    # apply_rope 需要 [b,s,d]，这里按行切回
    yy = y.clone()
    apply_rope(yy.view(1, S, -1)[..., -rot:], freqs)
    return yy

for (M, S, D, rot) in [(1, 1, 576, 64), (8, 8, 576, 64), (1, 1, 512, 64)]:
    x = torch.randn(M, D, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(D, dtype=torch.bfloat16, device="cuda")
    freqs = torch.randn(S, rot // 2, dtype=torch.complex64, device="cuda")
    freqs = freqs / freqs.abs()          # 单位复数，像真实 rotary
    eps = 1e-6

    y = rms_norm(x, w, eps)
    ref_out = y.clone()
    apply_rope(ref_out.view(1, S, -1)[..., -rot:], freqs)

    out = torch.empty_like(x)
    blk = min(D // 8, 1024 if M < 256 else 256)
    native.launch(native.kernel("rms_norm_rope", "rms_norm_rope_bf"),
                  (M,), (blk,),
                  [x, w, freqs.view(torch.float32).view(-1, 2), out, M, S, D, rot, float(eps)],
                  shared=32 * 4)
    torch.cuda.synchronize()
    d = (out.float() - ref_out.float()).abs().max().item()
    print(f"M={M} S={S} D={D} rot={rot}: maxdiff={d:.3e} bit={bool((out==ref_out).all())}")
