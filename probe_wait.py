"""8GB 上每一层的 80ms 到底在等什么：把 MoE 的 kernel 时间单独量出来。"""
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

# 用 CUDA event 分别包 attn 和 ffn，在真实 decode 里
import tinyinfer.deepseek.model as M
acc = {}
_orig = M.Block.forward
def fwd(self, *a, **k):
    lid = self.layer_id
    A, F = self.attn, self.ffn
    e0 = torch.cuda.Event(enable_timing=True); e0.record()
    # 手工复刻 Block.forward 的两段，中间插事件
    res = _orig(self, *a, **k)
    e1 = torch.cuda.Event(enable_timing=True); e1.record()
    acc.setdefault(lid, []).append((e0, e1))
    return res
M.Block.forward = fwd

# 单独包 ffn.forward
accf = {}
_omoe = M.MoE.forward if hasattr(M, "MoE") else None
from tinyinfer.deepseek.moe import MoE as _MoE
_ofn = _MoE.forward
def ffn_fwd(self, *a, **k):
    e0 = torch.cuda.Event(enable_timing=True); e0.record()
    r = _ofn(self, *a, **k)
    e1 = torch.cuda.Event(enable_timing=True); e1.record()
    accf.setdefault(self.layer_id, []).append((e0, e1))
    return r
_MoE.forward = ffn_fwd

acc.clear(); accf.clear()
torch.cuda.synchronize()
logits = e.forward(torch.tensor([[nxt]], dtype=torch.long, device="cuda"), pos)
torch.cuda.synchronize()
tot_blk = sum(acc[l][-1][0].elapsed_time(acc[l][-1][1]) for l in acc)
tot_moe = sum(accf[l][-1][0].elapsed_time(accf[l][-1][1]) for l in accf)
print("BLOCK total %.1f ms   MoE total %.1f ms   attn+rest %.1f ms" %
      (tot_blk, tot_moe, tot_blk - tot_moe), flush=True)
for l in sorted(acc)[:4]:
    b = acc[l][-1][0].elapsed_time(acc[l][-1][1])
    m = accf[l][-1][0].elapsed_time(accf[l][-1][1]) if l in accf else float("nan")
    print("  layer %d  block %.1f  moe %.1f  attn %.1f" % (l, b, m, b - m), flush=True)
