import sys, time, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel

torch.manual_seed(0)
eng = Engine(r"E:\Develop\Deepseek\tinyinfer\cache\weights", tier=OptLevel.BALANCED,
             device="cuda", layers=list(range(40)))
prompt = list(range(8))
with torch.inference_mode():
    logits = eng.forward(torch.tensor([prompt]), 0)
    nid = int(logits[0, -1].argmax())
    eng.forward(torch.tensor([[nid]]), 8)  # warm
    torch.cuda.synchronize()
    for i in eng.layer_ids:
        eng.block(i).profile = True
    eng.profile = True
    t0 = time.perf_counter()
    eng.forward(torch.tensor([[nid]]), 9)
    torch.cuda.synchronize()
    tot = (time.perf_counter() - t0) * 1e3
    parts = {i: dict(eng.block(i)._parts) for i in eng.layer_ids}
    print(f"token: {tot:.0f} ms")
    acc = {"hc_pre": 0.0, "attn": 0.0, "hc_mid": 0.0, "ffn": 0.0}
    on_gpu = 0
    for i, p in parts.items():
        for k in acc:
            acc[k] += p[k]
    print("sum:", {k: round(v, 1) for k, v in acc.items()})
    # per-layer totals
    layer_tot = {i: parts[i]["hc_pre"] + parts[i]["attn"] + parts[i]["hc_mid"] + parts[i]["ffn"] for i in parts}
    print("resident avg:", round(sum(layer_tot[i] for i in range(7)) / 7, 1))
    print("streamed avg:", round(sum(layer_tot[i] for i in range(7, 40)) / 33, 1))
