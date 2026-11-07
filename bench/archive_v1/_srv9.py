import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from tinyinfer.chat import encode_messages
from transformers import AutoTokenizer
import os
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
text = encode_messages([{'role':'user','content':'The capital of France is'}], thinking_mode='chat')
ids = list(tok.encode(text))
mode = os.environ.get('MODE','planner')
lay = None if mode == 'planner' else list(range(40))
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=lay)
with torch.inference_mode():
    eng.reset()
    logits = eng.forward(torch.tensor([ids], device='cuda'), 0)
    first = int(logits[0,-1].argmax())
print(f'RESULT mode={mode}: first={first} ({tok.decode([first])!r}) gpu_layers={eng.gpu_layers}')