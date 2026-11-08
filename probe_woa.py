import sys, time, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer

MODEL = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
e = Engine(MODEL, tier=OptLevel.AGGRESSIVE, device="cuda", layers=[0, 1], tokenizer=tok)
b0 = e._blocks[0]
b2 = e._blocks.get(2)
attn = b0.attn
print("attn type:", type(attn).__name__, flush=True)
for nm in ("wo_a", "wo_b", "wq_a", "wq_b", "wkv"):
    t = getattr(attn, nm, None)
    if t is not None:
        w = getattr(t, "weight", t)
        print("  %-6s %-16s %s" % (nm, str(tuple(w.shape)), w.dtype), flush=True)

# wo_a 反量化后的字节数
for nm in ("wo_a",):
    t = getattr(attn, nm, None)
    if t is not None and hasattr(t, "weight"):
        w = t.weight
        o, i = int(w.shape[0]), int(w.shape[1])
        print("  %s dequant bf16 = %.1f MB" % (nm, o * i * 2 / 2**20), flush=True)
        print("  x40 layers = %.2f GB" % (o * i * 2 * 40 / 2**30), flush=True)
