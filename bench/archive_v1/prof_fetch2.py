import sys, time, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
import tinyinfer.loading as L
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel

T = {"hit": 0, "miss": 0, "disk": 0.0, "copy_in": 0.0, "acquire": 0.0, "read_calls": 0}
orig_get = L.ExpertCache.get
def get(self, layer, expert, **kw):
    t0 = time.perf_counter()
    out = orig_get(self, layer, expert, **kw)
    dt = time.perf_counter() - t0
    if dt < 0.0005:
        T["hit"] += 1
    else:
        T["miss"] += 1
        T["disk"] += dt
    return out
L.ExpertCache.get = get
orig_read = L.ShardReader.read_blob
def read(self, blob):
    t0 = time.perf_counter()
    out = orig_read(self, blob)
    T["disk"] += time.perf_counter() - t0
    T["read_calls"] += 1
    return out
L.ShardReader.read_blob = read
orig_ci = L.VramSlotCache.copy_in
def ci(self, *a, **kw):
    t0 = time.perf_counter()
    out = orig_ci(self, *a, **kw)
    T["copy_in"] += time.perf_counter() - t0
    return out
L.VramSlotCache.copy_in = ci

eng = Engine(r"E:\Develop\Deepseek\tinyinfer\cache\weights", tier=OptLevel.BALANCED,
             device="cuda", layers=list(range(26)))
with torch.inference_mode():
    eng.forward(torch.tensor([[1, 2, 3, 4]]), 0)
    nid = int(eng.forward(torch.tensor([[1]]), 4)[0, -1].argmax())
    for k in T:
        T[k] = 0
    t0 = time.time()
    eng.forward(torch.tensor([[nid]]), 5)
    torch.cuda.synchronize()
    wall = time.time() - t0
print(f"wall {wall*1000:.0f}ms | hits={T['hit']} misses={T['miss']} | diskread {T['disk']*1000:.0f}ms (calls {T['read_calls']}) | copy_in {T['copy_in']*1000:.0f}ms")
