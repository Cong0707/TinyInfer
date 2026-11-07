import sys
class Block:
    def find_module(self, name, path=None):
        if name == 'tilelang' or name.startswith('tilelang.'):
            raise ImportError('blocked for test')
        return None
    def find_spec(self, name, path=None, target=None):
        if name == 'tilelang' or name.startswith('tilelang.'):
            raise ImportError('blocked for test')
        return None
sys.meta_path.insert(0, Block())
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
import torch, time
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
p = tok('Hello', return_tensors='pt').input_ids[0].tolist()
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
with torch.inference_mode():
    eng.reset(); t0=time.perf_counter()
    out = list(eng.stream_generate(p, max_new_tokens=2, temperature=0.0))
    dt = time.perf_counter()-t0
print(f'RESULT no-tilelang OK  {dt*1000:.0f}ms/2tok  out={out}')