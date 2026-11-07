import sys, torch, os, time
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
os.environ['TI_MTP'] = '1'
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
p = tok('The capital of', return_tensors='pt').input_ids[0].tolist()
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=[0, 1])
with torch.inference_mode():
    eng.reset()
    out = eng.speculative_generate(p, max_new_tokens=6, temperature=0.0)
    eng.reset()
    base = list(eng.stream_generate(p, max_new_tokens=6, temperature=0.0))
print('RESULT spec raw:', out)
print('RESULT base    :', base)
# 解析
if isinstance(out, tuple):
    toks, extra = out[0], out[1]
    print(f'RESULT parsed tokens={toks} extra={extra}')
    print(f'RESULT base tokens={base}')
    n = min(len(toks), len(base))
    same = sum(1 for i in range(n) if toks[i]==base[i])
    print(f'RESULT match {same}/{n}')