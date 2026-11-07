import sys, torch, time
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
p = tok('Hello', return_tensors='pt').input_ids[0].tolist()
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
with torch.inference_mode():
    eng.reset(); list(eng.stream_generate(p, max_new_tokens=2, temperature=0.0))
    torch.cuda.synchronize()
    # 抓一张统计表
    st = getattr(eng, 'stats', None)
    if st: print('stats keys:', list(st)[:10])
    # H2D 字节数
    import tinyinfer.loading as L
    for attr in dir(L):
        if 'byte' in attr.lower() or 'stat' in attr.lower():
            print('attr', attr)
    torch.cuda.synchronize()
    a = torch.cuda.memory_allocated()
    b = torch.cuda.memory_reserved()
    print(f'RESULT alloc={a/2**30:.2f}GB reserved={b/2**30:.2f}GB')