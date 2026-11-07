import sys, torch, time
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
d = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
p = tok("Hello", return_tensors="pt").input_ids[0].tolist()
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device="cuda", layers=list(range(40)))
def stats():
    try: return eng.expert_store.store.stats()
    except Exception:
        try: return eng.store.stats()
        except Exception as e: return {"err": str(e)}
with torch.inference_mode():
    eng.reset()
    list(eng.stream_generate(p, max_new_tokens=2, temperature=0.0))  # warm
    for step in range(4):
        a = stats()
        t0 = time.time()
        list(eng.stream_generate(p, max_new_tokens=1, temperature=0.0))
        dt = time.time() - t0
        b = stats()
        dr = b.get("stream_reads",0)-a.get("stream_reads",0)
        dg = b.get("stream_gb",0)-a.get("stream_gb",0)
        print(f"step{step}: {dt:.2f}s  reads={dr}  {dg*1024:.0f}MB  {dg*1024/max(dt,1e-9):.0f} MB/s", flush=True)
