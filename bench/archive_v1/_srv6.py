import sys, torch, time
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.adapters import open_adapter
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from tinyinfer.chat import encode_messages
from transformers import AutoTokenizer
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
text = encode_messages([{'role':'user','content':'The capital of France is'}], thinking_mode='chat')
ids = list(tok.encode(text))
def run(label, **kw):
    eng = Engine(**kw)
    with torch.inference_mode():
        eng.reset()
        out = list(eng.stream_generate(ids, max_new_tokens=5, temperature=0.0))
    print(f'RESULT {label}: {out} -> {tok.decode(out, skip_special_tokens=True)!r}')
    del eng; torch.cuda.empty_cache()
run('adapter + 40层', adapter=open_adapter(d), tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))