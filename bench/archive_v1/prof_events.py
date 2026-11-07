import sys, time, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel

torch.manual_seed(0)
eng = Engine(r"E:\Develop\Deepseek\tinyinfer\cache\weights", tier=OptLevel.BALANCED,
             device="cuda", layers=[0,1,2,3,4,5,6])
prompt = list(range(16))
with torch.inference_mode():
    logits = eng.forward(torch.tensor([prompt]), 0)
    nid = int(logits[0, -1].argmax())
    for p in (16, 17):  # warmup incl. M=1 shapes
        eng.forward(torch.tensor([[nid]]), p)
    torch.cuda.synchronize()

    ids = torch.tensor([[nid]], device="cuda")
    N = 3
    t_tot = 0.0
    parts = {"embed": 0.0, "head": 0.0}
    layer_parts = {i: {"attn": 0.0, "ffn": 0.0, "hc": 0.0, "total": 0.0} for i in eng.layer_ids}
    for step in range(N):
        ev = lambda: torch.cuda.Event(enable_timing=True)
        cache = {}
        def mark(name):
            e = ev(); e.record(); cache[name] = e
        def lap(end_name, start_name):
            e = ev(); e.record(); torch.cuda.synchronize(); return cache[start_name].elapsed_time(cache.get(end_name) or e)
        mark("t0")
        h = eng.embed[ids]
        h = h.unsqueeze(2).repeat(1, 1, 4, 1)
        mark("embed")
        pre_mix = torch.zeros(1, 1, 4, dtype=torch.float32, device="cuda"); pre_mix[:, :, 0] = 1.0
        end_prev = "embed"
        for i in eng.layer_ids:
            b = eng.block(i)
            h_ = b.hc_mixes(h, b.hc_attn_fn, b.hc_attn_scale, b.hc_attn_base)
            mark(f"hc1_{i}")
            hh = b.hc_pre(h, h_ and pre_mix or pre_mix)
            mark(f"pre_{i}")
            o = b.attn.forward(hh, 20 + step)
            mark(f"attn_{i}")
            h_ = b.hc_post(h, o, h_[1], h_[2])
            mark(f"posta_{i}")
            f_ = b.hc_mixes(h, b.hc_ffn_fn, b.hc_ffn_scale, b.hc_ffn_base)
            mark(f"hc2_{i}")
            hh = b.hc_pre(h, h_[0])
            mark(f"pre2_{i}")
            o = b.ffn.forward(hh)
            mark(f"ffn_{i}")
            h = b.hc_post(h, o, f_[1], f_[2])
            mark(f"postf_{i}")
            layer_parts[i]["total"] += lap(f"postf_{i}", f"posta_{i}") if False else 0
        h = (pre_mix.unsqueeze(-1) * h.float()).sum(dim=2).to(h.dtype)
        h = torch.nn.functional.rms_norm(h, (5120,), eng.out_norm, eng.eps) if False else h
        mark("collapse")
        from tinyinfer.nn import rms_norm
        h = rms_norm(h, eng.out_norm, eng.eps)
        logits = torch.nn.functional.linear(h, eng.head)
        mark("head")
        torch.cuda.synchronize()
        t_tot += cache["t0"].elapsed_time(cache["head"])
    print(f"avg token: {t_tot/N:.1f} ms")
