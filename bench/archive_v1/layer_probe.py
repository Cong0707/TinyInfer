import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer

tok_dir = r"E:\Develop\Deepseek\tinyinfer\cache\weights"
tok = AutoTokenizer.from_pretrained(tok_dir, trust_remote_code=True)
ids = tok("Hello, I am DeepSeek.", return_tensors="pt").input_ids[0].tolist()
for n in (26, 27, 28, 29, 30, 32):
    eng = Engine(tok_dir, tier=OptLevel.BALANCED, device="cuda", layers=list(range(n)))
    with torch.inference_mode():
        outs = list(eng.stream_generate(ids, max_new_tokens=8, temperature=0.0))
    print(f"n={n}: {tok.decode(outs)!r}", flush=True)
    del eng
    torch.cuda.empty_cache()
