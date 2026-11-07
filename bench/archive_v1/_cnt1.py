import sys, torch, time, collections
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from tinyinfer.loading import ExpertRecord
from transformers import AutoTokenizer
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
p = tok('Hello', return_tensors='pt').input_ids[0].tolist()
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
cnt = collections.Counter(); bytes_ = [0]
_orig = ExpertRecord.tensors_on
def patched(self, device):
    cnt['calls'] += 1
    try: bytes_[0] += int(self.buffer.numel())
    except Exception: bytes_[0] += 18800640
    return _orig(self, device)
ExpertRecord.tensors_on = patched
with torch.inference_mode():
    eng.reset(); list(eng.stream_generate(p, max_new_tokens=3, temperature=0.0))
    cnt.clear(); bytes_[0] = 0
    t0 = time.perf_counter()
    list(eng.stream_generate(p, max_new_tokens=1, temperature=0.0))
    dt = time.perf_counter() - t0
ExpertRecord.tensors_on = _orig
print(f'RESULT step={dt*1000:.0f}ms tensors_on={cnt["calls"]} bytes={bytes_[0]/1e9:.2f}GB h2d_ms={bytes_[0]/26.5e9*1000:.0f}')