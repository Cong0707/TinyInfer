"""不��批时，8GB 卡上能跑多大的 prefill（正确性的代价）。"""
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
M = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
base = "1+1 等于几？只回答数字。" * 40
enc = tok.encode(base, add_special_tokens=False)
N = int(sys.argv[1])
ids = torch.tensor([enc[:N]], device="cuda")
e = Engine(M, tier=OptLevel.AGGRESSIVE, device="cuda", layers=[0,1,2,3,4], tokenizer=tok)
e.reset()
try:
    lg = e._forward_chunk(ids, 0)
    torch.cuda.synchronize()
    f = lg.float()
    bad = torch.isnan(f).any(dim=-1)[0].nonzero().flatten().tolist()
    print("RESULT N=%3d bad=%s" % (N, bad), flush=True)
except torch.AcceleratorError as ex:
    print("RESULT N=%3d OOM" % N, flush=True)
except RuntimeError as ex:
    print("RESULT N=%3d ERR %s" % (N, str(ex)[:50]), flush=True)
