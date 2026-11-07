import sys, os, time, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from tinyinfer.chat import encode_messages
from transformers import AutoTokenizer

MODEL = r"E:\models\DeepSeek-V4.1-Flash-1017"
tier_name = os.environ.get("TIER", "aggressive").upper()
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
e = Engine(MODEL, tier=getattr(OptLevel, tier_name), device="cuda",
           layers=list(range(40)), tokenizer=tok)
q = "1+1 等于几？只回答数字。"
e.reset()
ids = torch.tensor([tok.encode(encode_messages([{"role": "user", "content": q}]),
                               add_special_tokens=False)],
                   dtype=torch.long, device="cuda")
logits = None
t0 = time.perf_counter()
for p in range(ids.shape[1]):
    logits = e.forward(ids[:, p:p + 1], p)
pf = time.perf_counter() - t0
sl = e._blocks[0].ffn._expert_store.slots
print("PREFILL %.1fs tokens=%d slots=%d loads=%d evict=%d" %
      (pf, ids.shape[1], sl.slots, sl.loads, sl.evictions), flush=True)
pos = ids.shape[1]
out = []
t0 = time.perf_counter()
n = 16
for i in range(n):
    lg = logits[0, -1].float()
    nxt = int(lg.argmax())
    if nxt == tok.eos_token_id:
        break
    out.append(nxt)
    logits = e.forward(torch.tensor([[nxt]], dtype=torch.long, device="cuda"), pos)
    pos += 1
dt = time.perf_counter() - t0
print("DECODE %.2f tok/s loads=%d evict=%d hits=%d bykey=%d" %
      (len(out) / max(dt, 1e-9), sl.loads, sl.evictions, sl.hits, len(sl._by_key)), flush=True)
print("OUT", repr(tok.decode(out)), flush=True)
