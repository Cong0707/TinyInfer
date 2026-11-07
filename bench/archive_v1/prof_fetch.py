import sys, time, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.loading import ExpertStore
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel

fetch_log = []
orig_fetch = ExpertStore.fetch
def timed_fetch(self, layer, experts, cache=True):
    t0 = time.perf_counter()
    out = orig_fetch(self, layer, experts, cache=cache)
    fetch_log.append((layer, time.perf_counter() - t0))
    return out
ExpertStore.fetch = timed_fetch

eng = Engine(r"E:\Develop\Deepseek\tinyinfer\cache\weights", tier=OptLevel.BALANCED,
             device="cuda", layers=list(range(26)))
with torch.inference_mode():
    eng.forward(torch.tensor([[1, 2, 3, 4]]), 0)
    nid = int(eng.forward(torch.tensor([[1]]), 4)[0, -1].argmax())
    fetch_log.clear()
    t0 = time.time()
    eng.forward(torch.tensor([[nid]]), 5)
    wall = time.time() - t0
    torch.cuda.synchronize()
print(f"token wall {wall*1000:.0f}ms  fetches={len(fetch_log)}  fetch_total={sum(t for _, t in fetch_log)*1000:.0f}ms  fetch_max={max(t for _, t in fetch_log)*1000:.1f}ms")
print("per-layer fetch ms:", {l: round(t * 1000, 1) for l, t in fetch_log})
