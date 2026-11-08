import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer

MODEL = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
e = Engine(MODEL, tier=OptLevel.AGGRESSIVE, device="cuda", layers=[2], tokenizer=tok)
b = e._blocks[2]
print("is_kv_source:", getattr(b.attn, "is_kv_source", "?"), flush=True)
print("ratio:", getattr(b.attn, "ratio", "?"), flush=True)
print("has compressor:", getattr(b.attn, "compressor", None) is not None, flush=True)
print("has indexer:", getattr(b.attn, "indexer", None) is not None, flush=True)
HC, D = e.cfg.hc_mult, e.cfg.dim
torch.manual_seed(0)
x1 = torch.randn(1, 1, HC, D, dtype=torch.bfloat16, device="cuda") * 0.02
from tinyinfer.nn import rms_norm
pm = torch.zeros(1, 1, HC, dtype=torch.float32, device="cuda"); pm[:, :, 0] = 1.0
h1 = rms_norm(b.hc_pre(x1, pm), b.attn_norm, e.cfg.rms_norm_eps)

for sub in (b.attn, getattr(b.attn, "compressor", None), getattr(b.attn, "indexer", None)):
    if sub is not None and hasattr(sub, "reset"):
        sub.reset()
from tinyinfer.deepseek.compressor import shared_attn as _sa
_sa.compress_kv = None; _sa.index_k = None; _sa.topk_idxs = None; _sa.candidates = None

P = 2
b.attn.forward(h1, P, positions=torch.tensor([P], device="cuda"))
ck = _sa.compress_kv
print("shared compress_kv:", None if ck is None else (tuple(ck.shape), float(ck.float().abs().max())), flush=True)
