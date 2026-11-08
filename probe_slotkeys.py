"""为什么 hits 永远是 0：槽位池里到底存着谁。"""
import sys, time, os, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from tinyinfer.chat import encode_messages
from transformers import AutoTokenizer

MODEL = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
e = Engine(MODEL, tier=OptLevel.AGGRESSIVE, device="cuda", layers=list(range(4)), tokenizer=tok)
q = "1+1 等于几？只回答数字。"
e.reset()
ids = torch.tensor([tok.encode(encode_messages([{"role": "user", "content": q}]),
                    add_special_tokens=False)], dtype=torch.long, device="cuda")
logits = None
for p in range(ids.shape[1]):
    logits = e.forward(ids[:, p:p+1], p)

sl = e.expert_store.slots
print("slots=%d bykey=%d loads=%d evict=%d hits=%d" %
      (sl.slots, len(sl._by_key), sl.loads, sl.evictions, sl.hits), flush=True)
print("keys:", sorted(sl._by_key.keys())[:20], flush=True)

# 走一步 decode，看每一步之后槽位里是谁
nxt = int(logits[0, -1].float().argmax())
pos = ids.shape[1]
for step in range(2):
    before = len(sl._by_key)
    logits = e.forward(torch.tensor([[nxt]], dtype=torch.long, device="cuda"), pos)
    msg = "step %d: bykey %d -> %d  loads=%d evict=%d hits=%d keys=%s" % (
        step, before, len(sl._by_key), sl.loads, sl.evictions, sl.hits,
        sorted(sl._by_key.keys())[:10])
    print(msg, flush=True)
    nxt = int(logits[0, -1].float().argmax())
    pos += 1
