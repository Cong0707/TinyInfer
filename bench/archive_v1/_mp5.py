import sys, torch, os
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
os.environ['TI_MTP'] = '1'
os.environ['TI_SPEC_DEBUG'] = '1'
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
p = tok('The capital of France is', return_tensors='pt').input_ids[0].tolist()
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(10)))
print('RESULT draft_len:', eng.cfg.dspark_block_size)
with torch.inference_mode():
    eng.reset()
    out = eng.speculative_generate(p, max_new_tokens=6, temperature=0.0)
print('RESULT out:', out)