"""topk 对专家数的影响（每进程一个 topk）。"""
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
from tinyinfer.nn import rms_norm
M = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
topk = int(sys.argv[1])
e = Engine(M, tier=OptLevel.AGGRESSIVE, device="cuda", layers=[0,1,2,3], tokenizer=tok)
b = e.block(0)
if topk != 6:
    b.ffn.gate.topk = topk
base = tok.encode("1+1 等于几？只回答数字。请仔细思考。", add_special_tokens=False)
ids = torch.tensor([base[:8]], device="cuda")
h = e.embed[ids.cpu()].to(e.device) if e.embed_on_cpu else e.embed[ids]
h = h.unsqueeze(2).repeat(1, 1, e.cfg.hc_mult, 1)
pm = torch.zeros(1, 8, e.cfg.hc_mult, dtype=torch.float32, device=h.device)
pm[:, :, 0] = 1.0
hn = rms_norm(b.hc_pre(h, pm), b.attn_norm, e.cfg.rms_norm_eps)
w, idx = b.ffn.gate.forward(hn.view(-1, e.cfg.dim), None)
torch.cuda.synchronize()
print("RESULT topk=%d idx=%s n_exp=%d rowsum_min=%.4f" % (
      topk, tuple(idx.shape), int(idx.numel()),
      float(w.float().sum(-1).min())), flush=True)
