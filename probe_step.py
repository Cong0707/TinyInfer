import sys, time, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from tinyinfer.chat import encode_messages
from transformers import AutoTokenizer

MODEL = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
e = Engine(MODEL, tier=OptLevel.AGGRESSIVE, device="cuda", layers=list(range(40)), tokenizer=tok)
q = "1+1 等于几？只回答数字。"
e.reset()
ids = torch.tensor([tok.encode(encode_messages([{"role": "user", "content": q}]),
                    add_special_tokens=False)], dtype=torch.long, device="cuda")
logits = None
for p in range(ids.shape[1]):
    logits = e.forward(ids[:, p:p+1], p)
pos = ids.shape[1]
# 逐步计时，看时间花在哪一层
nxt = int(logits[0, -1].float().argmax())
spans = []
for i in range(6):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    logits = e.forward(torch.tensor([[nxt]], dtype=torch.long, device="cuda"), pos)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) * 1e3
    spans.append(dt)
    nxt = int(logits[0, -1].float().argmax())
    pos += 1
print("STEP_MS " + " ".join("%.0f" % s for s in spans), flush=True)
prof = getattr(e, "_expert_store", None)
if prof is not None and getattr(prof, "slots", None) is not None:
    sl = prof.slots
    print("slot loads=%d evict=%d hits=%d bykey=%d" %
          (sl.loads, sl.evictions, sl.hits, len(sl._by_key)), flush=True)
