import sys, time, torch, os
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
nxt = int(logits[0, -1].float().argmax())

# 打点：包住 FlatStream 的 apply 和 wait
flat_apply = getattr(e, "_flat_apply", None)
flat_wait = getattr(e, "_flat_wait", None)
acc = {"apply": 0.0, "wait": 0.0, "n_apply": 0, "n_wait": 0}
if flat_apply is not None:
    def _apply(layer, buf_idx):
        t0 = time.perf_counter()
        flat_apply(layer, buf_idx)
        acc["apply"] += time.perf_counter() - t0
        acc["n_apply"] += 1
    e._flat_apply = _apply
if flat_wait is not None:
    def _wait(*a, **k):
        t0 = time.perf_counter()
        r = flat_wait(*a, **k)
        acc["wait"] += time.perf_counter() - t0
        acc["n_wait"] += 1
        return r
    e._flat_wait = _wait

torch.cuda.synchronize()
t0 = time.perf_counter()
logits = e.forward(torch.tensor([[nxt]], dtype=torch.long, device="cuda"), pos)
torch.cuda.synchronize()
total = (time.perf_counter() - t0) * 1e3
print("TOTAL_MS %.0f  apply=%.0fms(%d)  wait=%.0fms(%d)" %
      (total, acc["apply"]*1e3, acc["n_apply"], acc["wait"]*1e3, acc["n_wait"]), flush=True)
print("=> 剩余 %.0fms 在 block 内（attention/moe/hc）" % (total - (acc["apply"]+acc["wait"])*1e3), flush=True)
