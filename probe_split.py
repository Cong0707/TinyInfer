"""split a real layer into attn / ffn using CUDA events inside a live decode."""
import sys, time, os, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from tinyinfer.chat import encode_messages
from transformers import AutoTokenizer

MODEL = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
_NL = int(os.environ.get("TI_NLAYERS", "4"))
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

import tinyinfer.deepseek.model as M
import tinyinfer.deepseek.moe as MO
acc = {"attn": 0.0, "ffn": 0.0, "hc": 0.0, "n": 0}
_attn = None
_orig_blk = M.Block.forward
def fwd(self, *a, **k):
    A = getattr(self, "attn", None)
    F = getattr(self, "ffn", None)
    t0 = torch.cuda.Event(enable_timing=True); t0.record()
    # 只包 attn
    _a0 = torch.cuda.Event(enable_timing=True); _a0.record()
    _ow = A.forward
    def _af(*aa, **kk):
        return _ow(*aa, **kk)
    r = _orig_blk(self, *a, **k)
    t1 = torch.cuda.Event(enable_timing=True); t1.record()
    acc["n"] += 1
    acc.setdefault("evs", []).append((t0, t1))
    return r
M.Block.forward = fwd

# 直接逐个组件计时（在同一 stream 上背靠背）
b = e.block(0)
D, HC = e.cfg.dim, e.cfg.hc_mult
x4 = torch.randn(1, 1, HC, D, dtype=torch.bfloat16, device="cuda") * 0.02
pm = torch.zeros(1, 1, HC, dtype=torch.float32, device="cuda"); pm[:, :, 0] = 1.0
posv = torch.tensor([pos], device="cuda")

def ev_bench(fn, n=10):
    for _ in range(3): fn()
    torch.cuda.synchronize()
    e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(n): fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / n

# 在真实 decode 的那一层上量（随机输入会让 MoE 走空路由，量出来差两个数量级）
t_blk = ev_bench(lambda: _orig_blk(b, x4, pos, pm, positions=posv), n=3)
print("REAL_BLOCK_MS %.2f" % t_blk, flush=True)
t_moe = float("nan")
try:
    t_moe = ev_bench(lambda: b.ffn.forward(x4), n=3)
    print("REAL_FFN_MS %.2f" % t_moe, flush=True)
except Exception as exc:
    print("ffn bench skipped:", repr(exc)[:120], flush=True)
if t_moe == t_moe:
    print("ATTN+HC_MS %.2f" % (t_blk - t_moe), flush=True)
print("=> x40: block=%.0f ffn=%.0f attn=%.0f ms/token" %
      (t_blk*40, t_moe*40, (t_blk-t_moe)*40), flush=True)
