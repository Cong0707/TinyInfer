
import sys, time, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer

d = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
ids = tok("Hello, I am DeepSeek.", return_tensors="pt").input_ids[0].tolist()
eng = Engine(d, tier=OptLevel.BALANCED, device="cuda", layers=list(range(40)))

with torch.inference_mode():
    # 预热：跑 5 个 token 把 expert cache / CUDA 都填热
    t0 = time.time()
    list(eng.stream_generate(ids, max_new_tokens=5, temperature=0.0))
    print(f"[warm1] {time.time()-t0:.1f}s")

    # 第二轮：同一 prompt 再来一次，cache 全热
    for rnd in range(3):
        marks = []
        t0 = time.time()
        for _ in eng.stream_generate(ids, max_new_tokens=8, temperature=0.0):
            marks.append(time.time())
        gaps = sorted(marks[i]-marks[i-1] for i in range(1, len(marks)))
        med = gaps[len(gaps)//2]
        print(f"[round{rnd}] median {med*1000:.0f} ms/token -> {1/med:.2f} tok/s")
    # cache 统计
    st = getattr(eng, "expert_store", None) or getattr(eng, "store", None)
    if st is not None and hasattr(st, "stats"):
        print("cache stats:", st.stats())
