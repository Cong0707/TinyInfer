import sys, torch, time
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
p = tok('Hello', return_tensors='pt').input_ids[0].tolist()
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
# 给 _flat_apply 打点（层流式）
import tinyinfer.engine as E
tt = {'flat': 0.0, 'n': 0}
of = eng._flat_apply
def wf(layer, buf_idx):
    t0 = time.perf_counter(); r = of(layer, buf_idx)
    tt['flat'] += time.perf_counter()-t0; tt['n'] += 1
    return r
eng._flat_apply = wf
# 给 block.forward 打点
blocks = eng._blocks if hasattr(eng,'_blocks') else None
with torch.inference_mode():
    eng.reset(); list(eng.stream_generate(p, max_new_tokens=2, temperature=0.0))
    tt['flat']=0.0; tt['n']=0
    t0=time.perf_counter()
    list(eng.stream_generate(p, max_new_tokens=1, temperature=0.0))
    torch.cuda.synchronize()
    tot = time.perf_counter()-t0
print(f'RESULT total={tot*1000:.0f}ms flat_apply={tt["flat"]*1000:.0f}ms n={tt["n"]}')
print(f'RESULT flat_per_call={tt["flat"]/max(1,tt["n"])*1e6:.0f}us')