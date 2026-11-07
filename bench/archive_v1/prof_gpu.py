import sys, time, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel

torch.manual_seed(0)
eng = Engine(r"E:\Develop\Deepseek\tinyinfer\cache\weights", tier=OptLevel.BALANCED,
             device="cuda", layers=list(range(26)))
with torch.inference_mode():
    logits = eng.forward(torch.tensor([[1, 2, 3, 4]]), 0)
    nid = int(logits[0, -1].argmax())
    eng.forward(torch.tensor([[nid]]), 4)
    torch.cuda.synchronize()
    for i in eng.layer_ids:
        eng.block(i).profile = True
    eng.profile = True
    t0 = time.perf_counter()
    eng.forward(torch.tensor([[nid]]), 5)
    torch.cuda.synchronize()
    wall = (time.perf_counter() - t0) * 1000
    parts = {i: dict(eng.block(i)._parts) for i in eng.layer_ids}
gpu = sum(sum(p.values()) for p in parts.values())
print(f"wall {wall:.0f}ms  gpu_sum {gpu:.0f}ms  wait {wall-gpu:.0f}ms  gpu_util {gpu/wall*100:.0f}%")
acc = {"hc_pre": 0.0, "attn": 0.0, "hc_mid": 0.0, "ffn": 0.0}
for p in parts.values():
    for k in acc:
        acc[k] += p[k]
print("parts(ms):", {k: round(v, 1) for k, v in acc.items()})
res = sum(parts[i]["hc_pre"] + parts[i]["attn"] + parts[i]["hc_mid"] + parts[i]["ffn"] for i in range(7)) / 7
strm = sum(parts[i]["hc_pre"] + parts[i]["attn"] + parts[i]["hc_mid"] + parts[i]["ffn"] for i in range(7, 40)) / 33
print(f"per-layer: resident {res:.1f}ms  streamed {strm:.1f}ms")
