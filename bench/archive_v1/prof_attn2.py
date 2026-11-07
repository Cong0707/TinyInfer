import sys, time, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from tinyinfer.nn import rms_norm, apply_rope, window_idxs_general
from tinyinfer.kernels import sparse_attn_dispatch
from tinyinfer.quant import dequant_fp8_block

torch.manual_seed(0)
eng = Engine(r"E:\Develop\Deepseek\tinyinfer\cache\weights", tier=OptLevel.BALANCED,
             device="cuda", layers=[0, 1])
a = eng.block(0).attn
x = torch.randn(1, 1, 5120, dtype=torch.bfloat16, device="cuda") * 0.02
pos = 18
with torch.inference_mode():
    # prime state like decode step 18
    for p in range(1, pos + 1):
        a.forward(x, p)
    torch.cuda.synchronize()

    def ev_bench(fn, n=30):
        for _ in range(3): fn()
        torch.cuda.synchronize()
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(n): fn()
        e1.record()
        torch.cuda.synchronize()
        return e0.elapsed_time(e1) / n

    r = a
    xr = torch.randn(1, 1, r.wq_b.in_features, dtype=torch.bfloat16, device="cuda") * 0.02
    print("rms_norm        ", round(ev_bench(lambda: rms_norm(xr, r.q_norm, r.eps)), 3), "ms")
    qr = rms_norm(r.wq_a.forward(x), r.q_norm, r.eps)
    print("wq_a fp8 gemm   ", round(ev_bench(lambda: r.wq_a.forward(x)), 3), "ms")
    print("wq_b fp8 gemm   ", round(ev_bench(lambda: r.wq_b.forward(qr)), 3), "ms")
    q = r.wq_b.forward(qr).view(1, 1, r.n_heads, r.head_dim)
    freqs = r.freqs_cis[pos:pos+1]
    print("apply_rope      ", round(ev_bench(lambda: apply_rope(q[..., -r.rope_dim:], freqs)), 3), "ms")
    kv = rms_norm(r.wkv.forward(x), r.kv_norm, r.eps)
    old = r._kv_window
    kv_att = torch.cat([old, kv], 1)
    idxs = window_idxs_general(pos, 1, old.size(1), r.window, x.device).unsqueeze(0).expand(1, -1, -1)
    print("sparse_attn     ", round(ev_bench(lambda: sparse_attn_dispatch(q, kv_att, r.sink, idxs, r.scale)), 3), "ms")
    o = sparse_attn_dispatch(q, kv_att, r.sink, idxs, r.scale)
    print("wo_a dequant    ", round(ev_bench(lambda: dequant_fp8_block(r.wo_a.weight, r.wo_a.scale)), 3), "ms")
    wa = dequant_fp8_block(r.wo_a.weight, r.wo_a.scale).view(r.o_groups, r.o_lora_rank, -1)
    ov = o.view(1, 1, r.o_groups, -1)
    print("wo_a einsum     ", round(ev_bench(lambda: torch.einsum("bsgd,grd->bsgr", ov, wa)), 3), "ms")
    oe = torch.einsum("bsgd,grd->bsgr", ov, wa)
    print("wo_b fp8 gemm   ", round(ev_bench(lambda: r.wo_b.forward(oe.flatten(2))), 3), "ms")
    print("full attn       ", round(ev_bench(lambda: r.forward(x, pos)), 3), "ms")
