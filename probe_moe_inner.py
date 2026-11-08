"""MoE 内部 56ms 的构成：miss_scan 同步 / slot DMA / kernel 本身。"""
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

import tinyinfer.deepseek.moe as MO
acc = {"scan_ms": 0.0, "scan_n": 0, "gate_ms": 0.0, "gate_n": 0,
       "shared_ms": 0.0, "shared_n": 0, "fwd_ms": 0.0, "fwd_n": 0}
_scan = MO.MoE._miss_scan
_gate = MO.Gate.forward
def scan(self, *a, **k):
    t0 = time.perf_counter()
    r = _scan(self, *a, **k)
    acc["scan_ms"] += (time.perf_counter() - t0) * 1e3
    acc["scan_n"] += 1
    return r
MO.MoE._miss_scan = scan
def gate(self, *a, **k):
    t0 = time.perf_counter()
    r = _gate(self, *a, **k)
    acc["gate_ms"] += (time.perf_counter() - t0) * 1e3
    acc["gate_n"] += 1
    return r
MO.Gate.forward = gate

# 整个 MoE.forward 的 CUDA 时间 vs 主机时间
_fwd = MO.MoE.forward
busy = {"host": 0.0, "cuda": 0.0, "n": 0}
def fwd(self, *a, **k):
    h0 = time.perf_counter()
    e0 = torch.cuda.Event(enable_timing=True); e0.record()
    r = _fwd(self, *a, **k)
    e1 = torch.cuda.Event(enable_timing=True); e1.record()
    busy["host"] += (time.perf_counter() - h0) * 1e3
    busy["n"] += 1
    busy.setdefault("evs", []).append((e0, e1))
    return r
MO.MoE.forward = fwd

busy["evs"] = []
torch.cuda.synchronize()
t0 = time.perf_counter()
logits = e.forward(torch.tensor([[nxt]], dtype=torch.long, device="cuda"), pos)
torch.cuda.synchronize()
step = (time.perf_counter() - t0) * 1e3
busy["cuda"] = sum(a.elapsed_time(b) for a, b in busy["evs"])
print("STEP %.0fms  MoE host=%.0fms  MoE cuda=%.0fms  (n=%d)" %
      (step, busy["host"], busy["cuda"], busy["n"]), flush=True)
print("  miss_scan host=%.1fms n=%d  gate host=%.1fms n=%d" %
      (acc["scan_ms"], acc["scan_n"], acc["gate_ms"], acc["gate_n"]), flush=True)
