import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer

MODEL = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
e = Engine(MODEL, tier=OptLevel.AGGRESSIVE, device="cuda", layers=[0], tokenizer=tok)
blk = e._blocks[0]
sl = blk.ffn._expert_store.slots
print("slots=%d blob=%d" % (sl.slots, sl.blob_bytes), flush=True)

torch.manual_seed(0)
h = torch.randn(1, e.cfg.dim, dtype=torch.bfloat16, device="cuda") * 0.02
outs = []
for i in range(5):
    before = (len(sl._by_key), len(sl._free))
    y = blk.ffn.forward(h)
    outs.append(y.float().clone())
    keys = [sl._key[j] for j in range(sl.slots)]
    print("run %d absmax=%.4f bykey=%d/%d free=%d keys=%s" %
          (i, float(y.abs().max()), before[0], len(sl._by_key), len(sl._free), keys),
          flush=True)

ref = outs[0]
worst = max(float((o - ref).abs().max()) for o in outs[1:])
print("WORST_DIFF %.6f" % worst, flush=True)
print("loads=%d evict=%d hits=%d" % (sl.loads, sl.evictions, sl.hits), flush=True)
assert worst < 1e-3
print("MOE_SLOT_PATH_OK", flush=True)
