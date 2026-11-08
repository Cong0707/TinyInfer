"""decompose one decode step: flat-wait vs flat-apply(CPU submit) vs block compute."""
import sys, time, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from tinyinfer.chat import encode_messages
from transformers import AutoTokenizer

MODEL = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
import os
_NL = int(os.environ.get("TI_NLAYERS", "12"))
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

# 包住 _flat_apply 和 _flat_wait，记录主机时间
acc = {"apply_ms": 0.0, "apply_n": 0, "wait_ms": 0.0, "wait_n": 0}
_apply, _wait = e._flat_apply, e._flat_wait
def apply(layer, buf_idx):
    t0 = time.perf_counter()
    _apply(layer, buf_idx)
    acc["apply_ms"] += (time.perf_counter() - t0) * 1e3
    acc["apply_n"] += 1
def wait(layer, buf_idx):
    t0 = time.perf_counter()
    _wait(layer, buf_idx)
    acc["wait_ms"] += (time.perf_counter() - t0) * 1e3
    acc["wait_n"] += 1
e._flat_apply = apply
e._flat_wait = wait

for rep in range(3):
    acc.update(apply_ms=0.0, apply_n=0, wait_ms=0.0, wait_n=0)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    logits = e.forward(torch.tensor([[nxt]], dtype=torch.long, device="cuda"), pos)
    torch.cuda.synchronize()
    total = (time.perf_counter() - t0) * 1e3
    nxt = int(logits[0, -1].float().argmax())
    print("rep%d TOTAL=%.0fms  apply=%.0fms/%d  wait=%.0fms/%d  other=%.0fms" %
          (rep, total, acc["apply_ms"], acc["apply_n"], acc["wait_ms"], acc["wait_n"],
           total - acc["apply_ms"] - acc["wait_ms"]), flush=True)
