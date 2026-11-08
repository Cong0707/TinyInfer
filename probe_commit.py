"""commit 的 144ms：是 DMA 本身，还是每次建 Event 的开销。"""
import sys, time, os, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer

MODEL = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
e = Engine(MODEL, tier=OptLevel.AGGRESSIVE, device="cuda", layers=[0], tokenizer=tok)
st = e.expert_store
sl = st.slots
sl._ensure()
ram = st.ram
data = ram.get(0, 0)
if data is None:
    print("no ram data", flush=True)
    raise SystemExit

# 准备 24 个槽位
idxs = []
for i in range(min(14, sl.slots)):
    j = sl._free.pop() if sl._free else None
    if j is None:
        break
    idxs.append(j)
print("got %d slots" % len(idxs), flush=True)

# A: 只发 DMA，不建 Event
N = len(idxs)
torch.cuda.synchronize()
t0 = time.perf_counter()
from tinyinfer.loading import _async_h2d
with torch.cuda.stream(sl._copy_stream):
    for j in idxs:
        _async_h2d(sl._buffers[j], data, int(sl._buffers[j].numel()))
torch.cuda.synchronize()
tA = (time.perf_counter() - t0) * 1e3

# B: 加 Event
torch.cuda.synchronize()
t0 = time.perf_counter()
evs = []
with torch.cuda.stream(sl._copy_stream):
    for j in idxs:
        _async_h2d(sl._buffers[j], data, int(sl._buffers[j].numel()))
        ev = torch.cuda.Event()
        ev.record(sl._copy_stream)
        evs.append(ev)
torch.cuda.synchronize()
tB = (time.perf_counter() - t0) * 1e3

# C: 走完整 commit
torch.cuda.synchronize()
t0 = time.perf_counter()
torch.cuda.synchronize()
tC = (time.perf_counter() - t0) * 1e3
print("A: %d DMA no event = %.1f ms (%.2f ms each)" % (N, tA, tA / N), flush=True)
print("B: %d DMA with event = %.1f ms (%.2f ms each)" % (N, tB, tB / N), flush=True)
print("event overhead = %.1f ms for %d" % (tB - tA, N), flush=True)
