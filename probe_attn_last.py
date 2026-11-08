"""最后一个位置的 attention：sink / 有效槽位 / 输出。"""
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
M = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
base = "1+1 等于几？只回答数字。请仔细思考后再回答，不要着急。再加一些字凑够到三十个词左右吧。 "
enc = tok.encode(base, add_special_tokens=False)
ids = torch.tensor([enc[:11]], device="cuda")
e = Engine(M, tier=OptLevel.AGGRESSIVE, device="cuda", layers=[0,1,2,3,4], tokenizer=tok)
e.reset()

import tinyinfer.deepseek.compressed_attention as CA
import tinyinfer.kernels.sparse_attn as SA
orig_disp = SA.sparse_attn_dispatch
rec = []
def disp(q, kv, sink, idxs, scale, **kw):
    out = orig_disp(q, kv, sink, idxs, scale, **kw)
    torch.cuda.synchronize()
    # q/kv 形状 [b, m, h, d] / [b, n, d]
    qf = q.float(); kf = kv.float(); of = out.float()
    last = q.shape[1] - 1
    row_idx = idxs[0, last]
    valid = row_idx[row_idx >= 0]
    rec.append({
        "m": int(q.shape[1]), "n": int(kv.shape[1]), "k": int(idxs.shape[-1]),
        "valid": int(len(valid)), "sink_finite": bool(torch.isfinite(sink).all()),
        "sink_absmax": float(sink.abs().max()),
        "kv_finite": bool(torch.isfinite(kf).all()),
        "kv_nan": int(torch.isnan(kf).sum()),
        "q_last_finite": bool(torch.isfinite(qf[0, last]).all()),
        "q_last_absmax": float(qf[0, last].abs().max()),
        "out_last_finite": bool(torch.isfinite(of[0, last]).all()),
        "out_last_absmax": float(of[0, last].abs().max()) if torch.isfinite(of[0, last]).all() else None,
        "idx_max": int(row_idx.max()),
    })
    return out
SA.sparse_attn_dispatch = disp
CA.sparse_attn_dispatch = disp
import tinyinfer.deepseek.attention as AT
AT.sparse_attn_dispatch = disp

lg = e._forward_chunk(ids, 0)
f = lg.float()
print("logits bad positions:", torch.isnan(f).any(dim=-1)[0].nonzero().flatten().tolist())
for i, r in enumerate(rec):
    tag = "OK " if r["out_last_finite"] else "BAD"
    print("%s call%d m=%d n=%d valid=%d idx_max=%d kvnan=%d qmax=%.2f outmax=%s" % (
          tag, i, r["m"], r["n"], r["valid"], r["idx_max"], r["kv_nan"],
          r["q_last_absmax"], r["out_last_absmax"]), flush=True)
