
import sys, torch, time
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
import dataclasses

d = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
prompt = tok("The capital of France is", return_tensors="pt").input_ids[0].tolist()

for name, lvl, topk in (("AGG(top6)", OptLevel.AGGRESSIVE, 0),
                        ("AGG(top4)", OptLevel.AGGRESSIVE, 4),
                        ("AGG(top2)", OptLevel.AGGRESSIVE, 2)):
    eng = Engine(d, tier=lvl, device="cuda", layers=list(range(40)))
    if topk:
        eng.tier = dataclasses.replace(eng.tier, expert_topk=topk)
        eng._blocks.clear()
        for b in eng._blocks.values():
            getattr(b, "ffn", None) and setattr(b.ffn, "topk", topk)
    with torch.inference_mode():
        eng.reset()
        list(eng.stream_generate(prompt, max_new_tokens=2, temperature=0.0))
        t0 = time.perf_counter()
        out = list(eng.stream_generate(prompt, max_new_tokens=8, temperature=0.0))
        dt = time.perf_counter() - t0
    print(f"{name:10s} {dt/8*1000:6.0f} ms/tok  {8/dt:5.2f} tok/s  -> {repr(tok.decode(out)[:40])}")
    del eng
    torch.cuda.empty_cache()
