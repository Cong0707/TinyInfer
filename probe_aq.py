"""reset 后单行 decode：aq/asc 的内容是否正常。"""
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
from tinyinfer.deepseek.fp4_native import quantize_act
M = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
t = "1+1 等于几？只回答数字。请仔细思考后再回答。"
ids = torch.tensor([tok.encode(t, add_special_tokens=False)], device="cuda")
e = Engine(M, tier=OptLevel.AGGRESSIVE, device="cuda", layers=[0,1,2,3,4], tokenizer=tok)
moe = e.block(0).ffn

orig = moe._forward_gpu_only
def hook(x, xf, indices, weights):
    out = orig(x, xf, indices, weights)
    torch.cuda.synchronize()
    aq, asc = quantize_act(xf, xf.size(-1))
    rows = int(xf.size(0)); n_exp = int(indices.numel())
    tk = max(1, n_exp // rows)
    nk = asc.numel() // rows
    aqb = getattr(moe, "_aq_buf", None); ascb = getattr(moe, "_asc_buf", None)
    overlap_aq = 0 if aqb is None else int(aqb.numel()) - n_exp * int(xf.size(-1))
    overlap_asc = 0 if ascb is None else int(ascb.numel()) - n_exp * nk
    print("rows=%d n_exp=%d tk=%d nk=%d aq.numel=%d asc.numel=%d buf_aq=%s buf_asc=%s slack_aq=%d slack_asc=%d out_finite=%s" % (
          rows, n_exp, tk, nk, aq.numel(), asc.numel(),
          None if aqb is None else aqb.numel(), None if ascb is None else ascb.numel(),
          overlap_aq, overlap_asc, bool(torch.isfinite(out).all())), flush=True)
    return out
moe._forward_gpu_only = hook

e.reset()
e._forward_chunk(ids, 0)
torch.cuda.synchronize()
e.reset()
e._forward_chunk(ids[:, :1], 0)
torch.cuda.synchronize()
