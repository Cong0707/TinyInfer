"""RAM 缓存命中率：decode 时专家到底是从内存来还是从盘来。

每层 80ms 等 6 个专家 = 每个 13ms。18.8MB / 13ms = 1.4GB/s —— 那是磁盘速度。
如果命中 RAM 缓存，DMA 是 13.3GB/s = 1.4ms/个，6 个只要 8.4ms。
这个差就是命中率的差。
"""
import sys, time, os, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from tinyinfer.chat import encode_messages
from transformers import AutoTokenizer

MODEL = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
_NL = int(os.environ.get("TI_NLAYERS", "8"))
e = Engine(MODEL, tier=OptLevel.AGGRESSIVE, device="cuda", layers=list(range(_NL)), tokenizer=tok)
st = e.expert_store
ram = st.ram
print("RAM cache: capacity=%d entries, %.1f GB" % (
      getattr(ram, "capacity_entries", -1),
      getattr(ram, "capacity_entries", 0) * 18.8e6 / 2**30), flush=True)

q = "1+1 等于几？只回答数字。"
e.reset()
ids = torch.tensor([tok.encode(encode_messages([{"role": "user", "content": q}]),
                    add_special_tokens=False)], dtype=torch.long, device="cuda")
logits = None
for p in range(ids.shape[1]):
    logits = e.forward(ids[:, p:p+1], p)

h0 = getattr(ram, "_stats", None)
print("stats before:", h0, flush=True)

pos = ids.shape[1]
nxt = int(logits[0, -1].float().argmax())
for _ in range(2):
    logits = e.forward(torch.tensor([[nxt]], dtype=torch.long, device="cuda"), pos)
    nxt = int(logits[0, -1].float().argmax())
    pos += 1
torch.cuda.synchronize()

h1 = getattr(ram, "_stats", None)
print("stats after:", h1, flush=True)
if h0 is not None and h1 is not None:
    dh = h1.hits - h0.hits
    dm = h1.misses - h0.misses
    tot = dh + dm
    print("CACHE hits=%d misses=%d  hit_rate=%.1f%%" %
          (dh, dm, 100.0 * dh / max(1, tot)), flush=True)
print("read_blob=%s" % getattr(st, "_read_blobs", "?"), flush=True)
print("entries=%d" % len(getattr(ram, "_entries", {})), flush=True)
