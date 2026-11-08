"""prefill(17) -> reset -> 短输入(2 token)：是否必然 NaN。"""
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
M = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
e = Engine(M, tier=OptLevel.AGGRESSIVE, device="cuda", layers=[0,1,2,3], tokenizer=tok)

long_ids = torch.tensor([tok.encode("1+1 等于几？只回答数字。", add_special_tokens=False)], device="cuda")
short_ids = torch.tensor([tok.encode("你好", add_special_tokens=False)], device="cuda")
print("long=%d short=%d" % (long_ids.shape[1], short_ids.shape[1]), flush=True)

# A: 干净引擎跑短输入
e.reset()
la = e._forward_chunk(short_ids, 0)
torch.cuda.synchronize()
print("A clean short:      finite=%s" % bool(torch.isfinite(la).all()), flush=True)

# B: 长输入之后 reset，再短输入
e.reset()
e._forward_chunk(long_ids, 0)
torch.cuda.synchronize()
e.reset()
lb = e._forward_chunk(short_ids, 0)
torch.cuda.synchronize()
print("B long->reset->short: finite=%s" % bool(torch.isfinite(lb).all()), flush=True)

# C: 长输入之后不 reset，直接短输入
e.reset()
e._forward_chunk(long_ids, 0)
torch.cuda.synchronize()
lc = e._forward_chunk(short_ids, 0)
torch.cuda.synchronize()
print("C long->short(no reset): finite=%s" % bool(torch.isfinite(lc).all()), flush=True)
