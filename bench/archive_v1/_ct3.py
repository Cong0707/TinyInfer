import sys
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.chat import encode_messages
from transformers import AutoTokenizer
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
msgs = [{'role':'user','content':'The capital of France is'}]
for mode in ('chat', 'thinking'):
    out = encode_messages(msgs, thinking_mode=mode)
    ids = list(tok.encode(out))
    print(f'RESULT {mode}: rendered={out!r}')
    print(f'   ids({len(ids)})={ids}')