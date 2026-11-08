import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
import tinyinfer.deepseek.attention as AT

MODEL = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
e = Engine(MODEL, tier=OptLevel.AGGRESSIVE, device="cuda", layers=[0], tokenizer=tok)
b = e._blocks[0]
HC, D = e.cfg.hc_mult, e.cfg.dim
torch.manual_seed(0)
x1 = torch.randn(1, 1, HC, D, dtype=torch.bfloat16, device="cuda") * 0.02
from tinyinfer.nn import rms_norm
pm1 = torch.zeros(1, 1, HC, dtype=torch.float32, device="cuda"); pm1[:, :, 0] = 1.0
h1 = rms_norm(b.hc_pre(x1, pm1), b.attn_norm, e.cfg.rms_norm_eps)

capt = {}
_o = AT.sparse_attn_dispatch
def disp(q, kv, sink, idx, scale, **kw):
    capt.setdefault(q.shape[0], []).append(
        (q.float().clone(), kv.float().clone(), idx.clone()))
    return _o(q, kv, sink, idx, scale, **kw)
AT.sparse_attn_dispatch = disp

def fresh():
    b.attn.reset(); b.attn._kv_buf = None; b.attn._decode_idxs = None

fresh(); b.attn.forward(h1, 0, positions=torch.tensor([1], device="cuda"))
fresh(); b.attn.forward(torch.cat([h1,h1],0), 0, positions=torch.tensor([1,1], device="cuda"))

q1,kv1,idx1 = capt[1][-1]
q2,kv2,idx2 = capt[2][-1]
print("q    row0 vs single = %.6f" % float((q2[0:1] - q1).abs().max()), flush=True)
print("kv   row0 vs single = %.6f" % float((kv2[0:1] - kv1).abs().max()), flush=True)
print("idx  row0 vs single = %d" % int((idx2[0:1] != idx1).sum()), flush=True)
print("idx1 nonzero=%d  idx2 row0 nonzero=%d" % (int((idx1>=0).sum()), int((idx2[0:1]>=0).sum())), flush=True)
