import sys, time, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from tinyinfer.fused import fused_rmsnorm_rope, fused_rope_inplace
from tinyinfer.nn import rms_norm, apply_rope

torch.manual_seed(0)
eng = Engine(r"E:\Develop\Deepseek\tinyinfer\cache\weights", tier=OptLevel.BALANCED,
             device="cuda", layers=[0])
a = eng.block(0).attn
x = torch.randn(1, 1, 5120, dtype=torch.bfloat16, device="cuda") * 0.02
with torch.inference_mode():
    # reference: rms_norm(wq_a(x)) with q_norm  (no rope)
    qr_ref = rms_norm(a.wq_a.forward(x), a.q_norm, a.eps)
    qr_fused = fused_rmsnorm_rope(x, a.q_norm, None, None, rope_dim=0, eps=a.eps) if False else None
    # fused takes the projected tensor directly
    proj = a.wq_a.forward(x)  # [1,1,1280]
    qr_fused = fused_rmsnorm_rope(proj, a.q_norm, None, None, rope_dim=0, eps=a.eps)
    err = (qr_ref.float() - qr_fused.float()).abs().max().item()
    print("q_norm fused err:", err)

    # rope: compare apply_rope complex vs fused_rope_inplace
    def ev(fn, n=30):
        for _ in range(3): fn()
        torch.cuda.synchronize()
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(n): fn()
        e1.record(); torch.cuda.synchronize()
        return e0.elapsed_time(e1) / n

    pos = 20
    freqs = a.freqs_cis[pos:pos+1]
    cos = freqs.real.float().contiguous(); sin = freqs.imag.float().contiguous()
    q = torch.randn(1, 1, 64, 512, dtype=torch.bfloat16, device="cuda") * 0.02
    q_ref = q.clone(); apply_rope(q_ref[..., -64:], freqs)
    q_f = q.clone()
    fused_rope_inplace(q_f.reshape(-1, 512), cos, sin, 64, inverse=False)
    print("rope fused err:", (q_ref - q_f).abs().max().item())
    print("apply_rope torch ms:", round(ev(lambda: apply_rope(q.clone()[..., -64:], freqs)), 3))
    print("fused rope ms     :", round(ev(lambda: fused_rope_inplace(q_f.reshape(-1,512).clone(), cos, sin, 64)), 3))
    print("q_norm torch ms   :", round(ev(lambda: rms_norm(proj, a.q_norm, a.eps)), 3))
    print("q_norm fused ms   :", round(ev(lambda: fused_rmsnorm_rope(proj, a.q_norm, None, None, rope_dim=0, eps=a.eps)), 3))
