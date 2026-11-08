"""一层的 6 个专家在相邻 token 之间有多大重合？决定了能不能缓存复用。"""
import sys, torch
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

hist = {lid: [] for lid in range(4)}
pos = ids.shape[1]
nxt = int(logits[0, -1].float().argmax())
for step in range(8):
    logits = e.forward(torch.tensor([[nxt]], dtype=torch.long, device="cuda"), pos)
    for lid in range(4):
        b = e.block(lid)
        ids_ = getattr(b.ffn, "_last_ids", None)
        if ids_:
            hist[lid].append(tuple(sorted(ids_)))
    nxt = int(logits[0, -1].float().argmax())
    pos += 1

print("layer  overlap(t, t-1)  union over 8 steps" % (), flush=True)
tot_same = 0
tot_all = 0
for lid in range(4):
    h = hist[lid]
    if len(h) < 2:
        continue
    ovs = [len(set(h[i]) & set(h[i-1])) for i in range(1, len(h))]
    avg = sum(ovs) / len(ovs)
    union = set()
    for s in h:
        union |= set(s)
    tot_same += sum(ovs)
    tot_all += 6 * len(ovs)
    print("  L%d   avg %.1f/6 (%.0f%%)   union %d" % (lid, avg, 100*avg/6, len(union)), flush=True)
print("OVERALL overlap = %.0f%%" % (100.0 * tot_same / max(1, tot_all)), flush=True)
