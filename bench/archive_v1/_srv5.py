import sys, torch, time
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.adapters import open_adapter
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from tinyinfer.chat import encode_messages
from transformers import AutoTokenizer
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
adapter = open_adapter(d)
print('RESULT adapter:', type(adapter).__name__)
eng = Engine(adapter, tier=OptLevel.AGGRESSIVE, device='cuda', tokenizer=tok)
text = encode_messages([{'role':'user','content':'The capital of France is'}], thinking_mode='chat')
ids = list(tok.encode(text))
print('RESULT prompt:', ids)
with torch.inference_mode():
    eng.reset()
    t0 = time.perf_counter()
    out = list(eng.stream_generate(ids, max_new_tokens=5, temperature=0.0))
    torch.cuda.synchronize()
    dt = time.perf_counter()-t0
print(f'RESULT generated: {out}  ({dt:.1f}s)')
print('RESULT text:', repr(tok.decode(out, skip_special_tokens=True)))