import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer

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

import tinyinfer.kernels.sparse_attn as SA
seen = []
_orig = SA.sparse_attn_cuda
def traced(q, kv, sink, idx, scale):
    seen.append((tuple(q.shape), tuple(kv.shape), tuple(idx.shape),
                 int((idx < 0).sum())))
    return _orig(q, kv, sink, idx, scale)
SA.sparse_attn_cuda = traced
import tinyinfer.deepseek.attention as AT
_disp0 = AT.sparse_attn_dispatch
def _disp(q, kv, sink, idx, scale, **kw):
    seen.append((tuple(q.shape), tuple(kv.shape), tuple(idx.shape),
                 int((idx < 0).sum())))
    return _disp0(q, kv, sink, idx, scale, **kw)
AT.sparse_attn_dispatch = _disp

def fresh():
    b.attn.reset(); b.attn._kv_buf = None; b.attn._decode_idxs = None

seen.clear()
fresh(); o1 = b.attn.forward(h1, 0, positions=torch.tensor([1], device="cuda")).float()
print("single:", seen, flush=True)
seen.clear()
fresh(); o2 = b.attn.forward(torch.cat([h1,h1],0), 0, positions=torch.tensor([1,1], device="cuda")).float()
print("double:", seen, flush=True)
