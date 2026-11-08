"""同一专家在 batch 与单行两次运行里的槽位字节是否一致。"""
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
from tinyinfer.nn import rms_norm
M = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
e = Engine(M, tier=OptLevel.AGGRESSIVE, device="cuda", layers=[0], tokenizer=tok)
base = tok.encode("1+1 等于几？只回答数字。请仔细思考。", add_special_tokens=False)
N = 8
ids = torch.tensor([base[:N]], device="cuda")
b = e.block(0); moe = b.ffn
sl = e.expert_store.slots
h = e.embed[ids.cpu()].to(e.device) if e.embed_on_cpu else e.embed[ids]
h = h.unsqueeze(2).repeat(1, 1, e.cfg.hc_mult, 1)
pm = torch.zeros(1, N, e.cfg.hc_mult, dtype=torch.float32, device=h.device)
pm[:, :, 0] = 1.0
hn = rms_norm(b.hc_pre(h, pm), b.attn_norm, e.cfg.rms_norm_eps)
row = N - 1

def snap(experts):
    out = {}
    torch.cuda.synchronize()
    for ex in experts:
        k = (0, ex)
        s = sl._by_key.get(k)
        if s is not None:
            out[ex] = (s, sl.buffer(s).clone())
    return out

EXP = [52, 177, 200, 251, 286, 338]

e.reset()
moe.forward(hn)
torch.cuda.synchronize()
A = snap(EXP)
print("A slots: %s" % {k: v[0] for k, v in A.items()}, flush=True)

e.reset()
moe.forward(hn[:, row:row+1])
torch.cuda.synchronize()
B = snap(EXP)
print("B slots: %s" % {k: v[0] for k, v in B.items()}, flush=True)

for ex in EXP:
    if ex in A and ex in B:
        same = torch.equal(A[ex][1], B[ex][1])
        print("  expert %3d: slotA=%d slotB=%d bytes_equal=%s" % (
              ex, A[ex][0], B[ex][0], same), flush=True)
    else:
        print("  expert %3d: missing (A=%s B=%s)" % (ex, ex in A, ex in B), flush=True)
