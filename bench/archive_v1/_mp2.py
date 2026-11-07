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
print('RESULT mtp stages:', len(eng.mtp) if getattr(eng,'mtp',None) else 0)
with torch.inference_mode():
    eng.reset()
    # 基线：普通 generate
    t0 = time.perf_counter()
    base = list(eng.stream_generate(p, max_new_tokens=6, temperature=0.0))
    torch.cuda.synchronize()
    t_base = time.perf_counter() - t0
    eng.reset()
    # 投机
    t0 = time.perf_counter()
    spec = list(eng.speculative_generate(p, max_new_tokens=6, temperature=0.0))
    torch.cuda.synchronize()
    t_spec = time.perf_counter() - t0
print(f'RESULT base({len(base)} tok, {t_base*1000:.0f}ms): {base}')
print(f'RESULT spec({len(spec)} tok, {t_spec*1000:.0f}ms): {spec}')
n = min(len(base), len(spec))
if n:
    same = sum(1 for i in range(n) if base[i]==spec[i])
    print(f'RESULT first {n} tokens match: {same}/{n}')