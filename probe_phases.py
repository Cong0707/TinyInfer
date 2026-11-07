import sys, time, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer

MODEL = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
e = Engine(MODEL, tier=OptLevel.AGGRESSIVE, device="cuda", layers=[0], tokenizer=tok)
blk = e._blocks[0]
ffn = blk.ffn
sl = ffn._expert_store.slots
print("slots=%d" % sl.slots, flush=True)

torch.manual_seed(0)
h = torch.randn(1, e.cfg.dim, dtype=torch.bfloat16, device="cuda") * 0.02
# warm: first call loads the experts
for _ in range(6):
    ffn.forward(h)
torch.cuda.synchronize()

# phase 1: total (everything, GPU async)
n = 30
t0 = time.perf_counter()
for _ in range(n):
    ffn.forward(h)
torch.cuda.synchronize()
tot = (time.perf_counter() - t0) / n * 1e3

# phase 2: gate only
x = h.view(-1, e.cfg.dim)
t0 = time.perf_counter()
for _ in range(n):
    w, idx = ffn.gate.forward(x, None)
torch.cuda.synchronize()
gate = (time.perf_counter() - t0) / n * 1e3

# phase 3: gate + cpu sync (what the miss scan costs)
t0 = time.perf_counter()
for _ in range(n):
    w, idx = ffn.gate.forward(x, None)
    _ = int(idx.max())
sync = (time.perf_counter() - t0) / n * 1e3

# phase 4: miss scan (includes its own sync)
sids_dummy = torch.zeros(6, dtype=torch.int32, device="cuda")
t0 = time.perf_counter()
for _ in range(n):
    ffn._miss_scan(sids_dummy, idx.view(-1), 6)
scan = (time.perf_counter() - t0) / n * 1e3

print("PER_LAYER total=%.3fms gate=%.3fms gate+sync=%.3fms miss_scan=%.3fms" %
      (tot, gate, sync, scan), flush=True)
print("=> 40 layers: total %.1fms, scan %.1fms, sync %.1fms" %
      (tot * 40, scan * 40, (sync - gate) * 40), flush=True)
