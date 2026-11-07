import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
torch.cuda.synchronize()
print('RESULT === 显存分解 ===')
print('RESULT allocated: %.0f MB' % (torch.cuda.memory_allocated()/2**20))
print('RESULT reserved : %.0f MB' % (torch.cuda.memory_reserved()/2**20))
try:
    free, tot = torch.cuda.mem_get_info()
    print('RESULT free=%.0f MB total=%.0f MB' % (free/2**20, tot/2**20))
except Exception: pass
# 看 head/embed
for attr in ('head_w','embed_w','head','embed'):
    v = getattr(eng, attr, None)
    if v is not None:
        try: print('RESULT %s: %.1f MB' % (attr, v.numel()*v.element_size()/2**20))
        except Exception: pass