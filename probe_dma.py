"""RAM -> VRAM 的单一 DMA 到底多快。"""
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
print("slots=%d blob=%d" % (sl.slots, sl.blob_bytes), flush=True)

# 从 RAM 缓存取一个专家的字节
ram = st.ram
for e_ in range(20):
    pass
d = ram.get(0, 0)
if d is None:
    print("no ram entry for (0,0)", flush=True)
else:
    nb = len(d) if hasattr(d, "__len__") else d.nbytes
    print("ram blob nbytes=%d pinned=%s type=%s" % (nb, getattr(d, "is_pinned", lambda: "?")(), type(d).__name__), flush=True)

# 直接量 copy_in
idx = sl._claim_slot_locked()
print("claimed slot", idx, flush=True)
sl._key[idx] = (99, 0)
sl._by_key[(99, 0)] = idx
sl._state[idx] = sl.IN_USE
sl._refs[idx] = 1

N = 20
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(N):
    sl._copy_blocking(idx, d)
torch.cuda.synchronize()
dt = (time.perf_counter() - t0) / N
gbps = nb / dt / 1e9
print("COPY_IN %.1f ms/blob  = %.2f GB/s" % (dt * 1e3, gbps), flush=True)
print("=> 每步 24 个专家 = %.0f ms" % (dt * 24 * 1e3), flush=True)
