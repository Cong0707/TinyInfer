"""per-layer wall time inside a real decode step, measured with CUDA events."""
import sys, time, os, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from tinyinfer.chat import encode_messages
from transformers import AutoTokenizer

MODEL = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
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

# 用 CUDA event 包每一层的 forward（不要 perf_counter，CPU 时间不代表 GPU）
import tinyinfer.deepseek.model as M
evs = {}
_orig = M.Block.forward
def fwd(self, *a, **k):
    e0 = torch.cuda.Event(enable_timing=True)
    e0.record()
    r = _orig(self, *a, **k)
    e1 = torch.cuda.Event(enable_timing=True)
    e1.record()
    evs.setdefault(self.layer_id, []).append((e0, e1))
    return r
M.Block.forward = fwd

evs.clear()
torch.cuda.synchronize()
t0 = time.perf_counter()
logits = e.forward(torch.tensor([[nxt]], dtype=torch.long, device="cuda"), pos)
torch.cuda.synchronize()
total = (time.perf_counter() - t0) * 1e3
print("STEP_TOTAL_MS %.0f" % total, flush=True)
acc = 0.0
for lid in sorted(evs):
    e0, e1 = evs[lid][-1]
    ms = e0.elapsed_time(e1)
    acc += ms
    print("  layer %2d  %.2f ms" % (lid, ms), flush=True)
print("SUM_BLOCK_MS %.1f   UNACCOUNTED %.0f" % (acc, total - acc), flush=True)
