"""MoE buffer 在 1 行输入时的形状是否自洽。"""
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
M = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
t = "1+1 等于几？只回答数字。请仔细思考后再回答。"
ids = torch.tensor([tok.encode(t, add_special_tokens=False)], device="cuda")
e = Engine(M, tier=OptLevel.AGGRESSIVE, device="cuda", layers=[0,1,2,3,4], tokenizer=tok)
moe = e.block(0).ffn

orig = moe._forward_gpu_only
def hook(x, xf, indices, weights):
    rows = int(xf.size(0)); n_exp = int(indices.numel())
    tk = max(1, n_exp // rows)
    aqb = getattr(moe, "_aq_buf", None)
    ascb = getattr(moe, "_asc_buf", None)
    print("rows=%d n_exp=%d tk=%d | aq_buf=%s asc_buf=%s | aq.numel=%d asc.numel=%d" % (
          rows, n_exp, tk,
          None if aqb is None else aqb.numel(),
          None if ascb is None else ascb.numel(),
          int(0), int(0)), flush=True)
    return orig(x, xf, indices, weights)
moe._forward_gpu_only = hook

e.reset()
e._forward_chunk(ids, 0)
torch.cuda.synchronize()
print("--- after reset, decode ---", flush=True)
e.reset()
lg = e._forward_chunk(ids[:, :1], 0)
torch.cuda.synchronize()
print("decode finite=%s" % bool(torch.isfinite(lg).all()), flush=True)
print("final aq_buf=%s asc_buf=%s" % (
      None if getattr(moe,"_aq_buf",None) is None else moe._aq_buf.numel(),
      None if getattr(moe,"_asc_buf",None) is None else moe._asc_buf.numel()), flush=True)
