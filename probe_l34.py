"""L3 vs L4：compress_len 和 compress_kv 的实际状态。"""
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
M = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
ids = torch.tensor([tok.encode("1+1 等于几？只回答数字。", add_special_tokens=False)], device="cuda")
e = Engine(M, tier=OptLevel.AGGRESSIVE, device="cuda", layers=[0,1,2,3,4], tokenizer=tok)

# 拦截 MLACompressedAttention.forward 的入口与出口
import tinyinfer.deepseek.compressed_attention as CA
orig = CA.MLACompressedAttention.forward
rec = []
def hooked(self, x, start_pos=0, positions=None):
    lid = getattr(self, "layer_id", "?")
    rt = getattr(self, "rt", None)
    before = {}
    for k in ("compress_kv", "topk_idxs", "index_k", "candidates"):
        v = getattr(rt, k, None) if rt is not None else None
        if torch.is_tensor(v):
            before[k] = (tuple(v.shape), bool(torch.isfinite(v.float()).all()))
        else:
            before[k] = (None, None)
    out = orig(self, x, start_pos, positions)
    torch.cuda.synchronize()
    after = {}
    for k in ("compress_kv", "topk_idxs", "index_k", "candidates"):
        v = getattr(rt, k, None) if rt is not None else None
        if torch.is_tensor(v):
            after[k] = (tuple(v.shape), bool(torch.isfinite(v.float()).all()))
        else:
            after[k] = (None, None)
    rec.append((lid, bool(torch.isfinite(out).all()), float(out.float().abs().max()),
                float(x.float().abs().max()), before, after))
    return out
CA.MLACompressedAttention.forward = hooked

e.reset()
lg = e._forward_chunk(ids, 0)
torch.cuda.synchronize()
for lid, ok, am, xm, b, a in rec:
    print("L%s out finite=%-5s absmax=%9.3f  x_absmax=%9.3f" % (lid, ok, am, xm), flush=True)
    for k in ("compress_kv", "topk_idxs"):
        print("   %-12s before=%s after=%s" % (k, b.get(k), a.get(k)), flush=True)
