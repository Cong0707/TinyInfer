import sys, torch, os
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
p = tok('Hello', return_tensors='pt').input_ids[0].tolist()
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
# 抓第一层的 gate 两次调用结果对比
import tinyinfer.deepseek.moe as M
orig = M.Gate.forward
calls = []
def patched(self, x, image_mask=None):
    r = orig(self, x, image_mask)
    if len(calls) < 2:
        calls.append((x.detach().float().cpu().numpy().copy(), r[0].detach().cpu().numpy().copy(), r[1].detach().cpu().numpy().copy()))
    return r
M.Gate.forward = patched
with torch.inference_mode():
    eng.reset(); list(eng.stream_generate(p, max_new_tokens=1, temperature=0.0))
if len(calls) >= 2:
    print('RESULT layer0 call1 idx', calls[0][2].flatten().tolist())
    print('RESULT layer0 call2 idx', calls[1][2].flatten().tolist())
    print('RESULT same x?', (calls[0][0] == calls[1][0]).all())
else:
    print('RESULT calls', len(calls))