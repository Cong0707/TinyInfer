"""关掉 MoE 分批（阈值调大），看 p9 是否恢复。"""
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer

THRESH = int(sys.argv[1])
import tinyinfer.deepseek.moe as MOE
orig = MOE.MoE._forward_gpu_only
moe_mod = sys.modules["tinyinfer.deepseek.moe"]
# 用源码打补丁：把 xf.size(0) > 3 改成阈值
import inspect, textwrap
src = inspect.getsource(orig)

M = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
base = "1+1 等于几？只回答数字。请仔细思考后再回答，不要着急。再加一些字凑够到三十个词左右吧。 "
enc = tok.encode(base, add_special_tokens=False)
ids = torch.tensor([enc[:11]], device="cuda")
e = Engine(M, tier=OptLevel.AGGRESSIVE, device="cuda", layers=[0,1,2,3,4], tokenizer=tok)

# 直接把 Module 类里的阈值改掉：monkeypatch 分批入口
import types
new_src = src.replace("if xf.size(0) > 3:", "if xf.size(0) > %d:" % THRESH)
ns = dict(moe_mod.__dict__)
exec(textwrap.dedent(new_src), ns)
patched = ns["_forward_gpu_only"]
MOE.MoE._forward_gpu_only = patched

e.reset()
lg = e._forward_chunk(ids, 0)
torch.cuda.synchronize()
f = lg.float()
bad = torch.isnan(f).any(dim=-1)[0].nonzero().flatten().tolist()
print("RESULT thresh=%d bad=%s total_nan=%d" % (THRESH, bad, int(torch.isnan(f).sum())), flush=True)
