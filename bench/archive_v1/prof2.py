import sys, time, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel

torch.manual_seed(0)
eng = Engine(r"E:\Develop\Deepseek\tinyinfer\cache\weights", tier=OptLevel.BALANCED,
             device="cuda", layers=[0,1,2,3,4])
prompt = list(range(16))
with torch.inference_mode():
    logits = eng.forward(torch.tensor([prompt]), 0)
    nid = int(logits[0, -1].argmax())
    for p in (16, 17):
        eng.forward(torch.tensor([[nid]]), p)
    torch.cuda.synchronize()
    for i in eng.layer_ids:
        eng.block(i).profile = True
    eng.profile = True
    t0 = time.perf_counter()
    for p in (18, 19):
        eng.forward(torch.tensor([[nid]]), p)
        timings = list(eng._timings)
        parts = {i: dict(eng.block(i)._parts) for i in eng.layer_ids}
    torch.cuda.synchronize()
    print(f"total: {(time.perf_counter()-t0)/2*1e3:.1f} ms/token")
    print(f"{'L':>3} {'hc_pre':>6} {'attn':>6} {'hc_mid':>6} {'ffn':>6} {'hc_post':>7} {'total':>6}")
    for i, ms in timings:
        p = parts[i]
        print(f"{i:3d} {p['hc_pre']:6.2f} {p['attn']:6.2f} {p['hc_mid']:6.2f} {p['ffn']:6.2f} {p['hc_post']:7.2f} {ms:6.1f}")
