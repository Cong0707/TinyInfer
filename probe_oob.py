"""reset 后 decode：sparse_attn 收到的 kv 和 idxs 是否越界。"""
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
import tinyinfer.kernels.sparse_attn as SA
import tinyinfer.deepseek.attention as AT
import tinyinfer.deepseek.compressed_attention as CA
M = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
t = "1+1 等于几？只回答数字。请仔细思考后再回答。"
ids = torch.tensor([tok.encode(t, add_special_tokens=False)], device="cuda")
e = Engine(M, tier=OptLevel.AGGRESSIVE, device="cuda", layers=[0,1,2,3,4], tokenizer=tok)

orig = SA.sparse_attn_dispatch
rec = []
def disp(q, kv, sink, idxs, scale, **kw):
    # 检查越界
    n = int(kv.shape[1])
    mx = int(idxs.max()) if idxs.numel() else -1
    ok = mx < n
    out = orig(q, kv, sink, idxs, scale, **kw)
    out_ok = bool(torch.isfinite(out).all())
    rec.append((n, mx, ok, out_ok, int((idxs >= 0).sum())))
    return out
SA.sparse_attn_dispatch = disp
AT.sparse_attn_dispatch = disp
CA.sparse_attn_dispatch = disp

e.reset()
e._forward_chunk(ids, 0)
torch.cuda.synchronize()
e.reset()
rec.clear()
lg = e._forward_chunk(ids[:, :1], 0)
torch.cuda.synchronize()
f = lg.float()
print("decode finite=%s" % bool(torch.isfinite(f).all()), flush=True)
for i, (n, mx, ok, ook, nvalid) in enumerate(rec):
    print("  call%d kv_n=%3d idx_max=%3d in_bounds=%-5s out_finite=%-5s nvalid=%d" %
          (i, n, mx, ok, ook, nvalid), flush=True)
