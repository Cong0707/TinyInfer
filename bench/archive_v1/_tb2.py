import sys, traceback
class Block:
    def find_spec(self, name, path=None, target=None):
        if name == 'tilelang' or name.startswith('tilelang.'):
            raise ImportError('blocked')
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
try:
    eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
    print('ENGINE BUILT')
    with torch.inference_mode():
        eng.reset()
        out = list(eng.stream_generate(p, max_new_tokens=2, temperature=0.0))
    print('RUN OK', out)
except Exception:
    print('=== FAILED ===')
    for line in traceback.format_exc().strip().splitlines()[-18:]:
        print(line)