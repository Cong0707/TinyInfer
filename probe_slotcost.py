"""56ms/层 是 GPU 在执行，还是 GPU 在等 DMA？用 wait_ready 的耗时切分。"""
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
q = "1+1 等于几？只回答数字。"
e.reset()
ids = torch.tensor([tok.encode(encode_messages([{"role":"user","content":q}]),
                    add_special_tokens=False)], dtype=torch.long, device="cuda")
logits = None
for p in range(ids.shape[1]):
    logits = e.forward(ids[:, p:p+1], p)
pos = ids.shape[1]
nxt = int(logits[0, -1].float().argmax())

st = e.expert_store
sl = st.slots
counters = {"need": 0, "need_ms": 0.0, "commit": 0, "commit_ms": 0.0,
            "hits": 0, "miss": 0}
_need = sl.stage
_commit = sl.commit
def stage(layer, experts, loader, urgent=True):
    t0 = time.perf_counter()
    r = _need(layer, experts, loader, urgent=urgent)
    counters["need_ms"] += (time.perf_counter() - t0) * 1e3
    counters["need"] += 1
    counters["miss"] += int(r)
    return r
sl.stage = stage
def commit(layer=None):
    t0 = time.perf_counter()
    r = _commit(layer)
    counters["commit_ms"] += (time.perf_counter() - t0) * 1e3
    counters["commit"] += 1
    return r
sl.commit = commit

torch.cuda.synchronize()
t0 = time.perf_counter()
logits = e.forward(torch.tensor([[nxt]], dtype=torch.long, device="cuda"), pos)
torch.cuda.synchronize()
step = (time.perf_counter() - t0) * 1e3
print("STEP %.0fms   stage=%.1fms/%d (placed %d)   commit=%.1fms/%d" %
      (step, counters["need_ms"], counters["need"], counters["miss"],
       counters["commit_ms"], counters["commit"]), flush=True)
print("slot loads=%d evict=%d hits=%d bykey=%d" %
      (sl.loads, sl.evictions, sl.hits, len(sl._by_key)), flush=True)
