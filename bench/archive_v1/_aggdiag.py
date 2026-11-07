
import sys, time, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer

d = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
ids = tok("Hello, I am DeepSeek.", return_tensors="pt").input_ids[0].tolist()
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device="cuda", layers=list(range(40)))
hp = getattr(eng, "host_pool", None)
print("HOSTPOOL GB:", round(hp.total/2**30, 1) if hp else 0)
st = getattr(eng, "expert_store", None)
if st is not None:
    for attr in ("cache_bytes", "capacity", "_cache"):
        v = getattr(st, attr, None)
        if v is not None:
            print("store." + attr, v if isinstance(v, int) else type(v).__name__)
with torch.inference_mode():
    list(eng.stream_generate(ids, max_new_tokens=3, temperature=0.0))
    marks = []
    for _ in eng.stream_generate(ids, max_new_tokens=10, temperature=0.0):
        marks.append(time.time())
gaps = sorted(marks[i]-marks[i-1] for i in range(2, len(marks)))
med = gaps[len(gaps)//2]
print(f"AGG: {med*1000:.0f} ms/token -> {1/med:.2f} tok/s")
try:
    print("STATS:", st.stats() if hasattr(st, "stats") else "n/a")
except Exception as e:
    print("STATS err:", e)
