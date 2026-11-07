import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from tinyinfer.chat import encode_messages
from transformers import AutoTokenizer
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
text = encode_messages([{'role':'user','content':'The capital of France is'}], thinking_mode='chat')
ids = list(tok.encode(text))
print('RESULT prompt ids:', ids)
with torch.inference_mode():
    eng.reset()
    logits = eng.forward(torch.tensor([ids], device='cuda'), 0)
    nxt = int(logits[0,-1].argmax())
    print(f'RESULT first token: {nxt} -> {tok.decode([nxt])!r}')
    # 完整跑 8 个
    eng.reset()
    out = list(eng.stream_generate(ids, max_new_tokens=8, temperature=0.0))
    print('RESULT generated:', out)
    print('RESULT text:', repr(tok.decode(out, skip_special_tokens=True)))