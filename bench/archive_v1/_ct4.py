import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from tinyinfer.chat import encode_messages
from transformers import AutoTokenizer
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
for nlayers, label in ((40,'40层'),):
    text = encode_messages([{'role':'user','content':'The capital of France is'}], thinking_mode='chat')
    ids = list(tok.encode(text))
    with torch.inference_mode():
        eng.reset()
        out = list(eng.stream_generate(ids, max_new_tokens=8, temperature=0.0))
    print(f'RESULT {label}: {out}')
    print(f'RESULT decoded: {tok.decode(out)!r}')