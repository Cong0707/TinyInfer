import sys, time, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from tinyinfer.routes import RoutePredictor

torch.manual_seed(0)
eng = Engine(r"E:\Develop\Deepseek\tinyinfer\cache\weights", tier=OptLevel.BALANCED,
             device="cuda", layers=list(range(5)))
eng.predictor = None
eng.enable_prefetch = False

pred = RoutePredictor()

prompt = [19923, 14, 342, 1030, 22651, 4374, 1465, 16, 17, 18, 19, 20, 21, 22, 23, 24]
with torch.inference_mode():
    logits = eng.forward(torch.tensor([prompt]), 0)
    nid = int(logits[0, -1].argmax())
    prev = dict(eng._last_routing)
    temporal_hits = temporal_tot = 0
    N = 40
    t0 = time.perf_counter()
    for step in range(N):
        p = len(prompt) + step
        prediction = pred.predict(nid)
        logits = eng.forward(torch.tensor([[nid]]), p)
        cur = dict(eng._last_routing)
        pred.observe(nid, cur)
        nid = int(logits[0, -1].argmax())
        for layer, ids in cur.items():
            if layer in prev:
                temporal_tot += 1
                temporal_hits += int(bool(set(ids) & set(prev[layer])))
        prev = cur
        if step == N // 2:
            print("mid:", pred.stats())
    dt = (time.perf_counter() - t0) / N * 1e3
print(f"decode {dt:.1f} ms/token (5 layers)")
print("predictor:", pred.stats())
print(f"temporal baseline: {temporal_hits/max(temporal_tot,1):.3f} over {temporal_tot}")
