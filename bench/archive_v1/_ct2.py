import sys
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.chat import encode_messages
msgs = [{'role':'user','content':'The capital of France is'}]
out = encode_messages(msgs, thinking_mode=False)
print('RESULT rendered:', repr(out))
from transformers import AutoTokenizer
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
ids = list(tok.encode(out))
print(f'RESULT ids ({len(ids)}):', ids)
print('RESULT decoded back:', repr(tok.decode(ids)))