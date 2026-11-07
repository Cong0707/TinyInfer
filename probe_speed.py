import sys, time, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer

MODEL = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
e = Engine(MODEL, tier=OptLevel.AGGRESSIVE, device="cuda", layers=[0], tokenizer=tok)
blk = e._blocks[0]
sl = blk.ffn._expert_store.slots
print("slots=%d" % sl.slots, flush=True)

torch.manual_seed(0)
h = torch.randn(1, e.cfg.dim, dtype=torch.bfloat16, device="cuda") * 0.02
for _ in range(5):
    blk.ffn.forward(h)
torch.cuda.synchronize()

n = 50
t0 = time.perf_counter()
for _ in range(n):
    blk.ffn.forward(h)
torch.cuda.synchronize()
dt = time.perf_counter() - t0
print("MOE_LAYER_MS %.2f  (%.1f layers/token -> %.1f ms/token)" %
      (dt / n * 1e3, 40, dt / n * 1e3 * 40), flush=True)
print("loads=%d evict=%d hits=%d bykey=%d" %
      (sl.loads, sl.evictions, sl.hits, len(sl._by_key)), flush=True)
