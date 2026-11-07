import sys, torch, os
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
os.environ['TI_MTP'] = '1'
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
p = tok('Hi', return_tensors='pt').input_ids[0].tolist()
# 只用 2 层，省显存给 draft
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=[0, 1])
print('RESULT layers:', len([b for b in eng._blocks.values()]) if hasattr(eng,'_blocks') else '?')
print('RESULT mtp:', len(eng.mtp) if getattr(eng,'mtp',None) else 0)
print('RESULT vram used:', torch.cuda.memory_allocated()/2**30, 'GB')