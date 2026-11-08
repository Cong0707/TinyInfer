import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
M = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
e = Engine(M, tier=OptLevel.AGGRESSIVE, device="cuda", layers=[0,1,2,3], tokenizer=tok)
base = tok.encode("1+1 等于几？只回答数字。请仔细思考。", add_special_tokens=False)
N = int(sys.argv[1])
ids = torch.tensor([base[:N]], device="cuda")
print("N=%d ids=%s" % (N, tuple(ids.shape)), flush=True)
e.reset()
lg = e._forward_chunk(ids, 0)
torch.cuda.synchronize()
f = lg.float()
print("  batch finite=%s absmax=%.3f" % (bool(torch.isfinite(f).all()),
      float(f.abs().max()) if torch.isfinite(f).all() else -1), flush=True)
e.reset()
last = None
for p in range(N):
    last = e._forward_chunk(ids[:, p:p+1], p)[0, 0].float()
torch.cuda.synchronize()
print("  token finite=%s absmax=%.3f" % (bool(torch.isfinite(last).all()),
      float(last.abs().max()) if torch.isfinite(last).all() else -1), flush=True)
