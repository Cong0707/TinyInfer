"""1024 token 时压缩 attention 的完整堆栈。"""
import sys, traceback, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
from tinyinfer.nn import rms_norm
M = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
e = Engine(M, tier=OptLevel.AGGRESSIVE, device="cuda", layers=[2], tokenizer=tok)
b = e.block(2)
base = tok.encode("你好今天天气不错", add_special_tokens=False)
ids = torch.tensor([(base*300)[:1024]], device="cuda")
e.reset()

h = e.embed[ids.cpu()].to(e.device) if e.embed_on_cpu else e.embed[ids]
h = h.unsqueeze(2).repeat(1, 1, e.cfg.hc_mult, 1)
pm = torch.zeros(1, 1024, e.cfg.hc_mult, dtype=torch.float32, device=h.device)
pm[:, :, 0] = 1.0
apre, apost, acomb = b.hc_mixes(h.float(), b.hc_attn_fn, b.hc_attn_scale, b.hc_attn_base)
hn = rms_norm(b.hc_pre(h, pm), b.attn_norm, e.cfg.rms_norm_eps)
pos = torch.arange(1024, device="cuda").unsqueeze(0)
try:
    ya = b.attn.forward(hn, 0, positions=pos)
    torch.cuda.synchronize()
    print("OK finite=%s" % bool(torch.isfinite(ya).all()), flush=True)
except Exception:
    traceback.print_exc()
