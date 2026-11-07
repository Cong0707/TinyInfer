import sys
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.chat import encode_messages
from transformers import AutoTokenizer
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
text = encode_messages([{'role':'user','content':'The capital of France is'}], thinking_mode='chat')
print('RESULT text:', repr(text))
# 逐 token 解码看结构
ids = list(tok.encode(text))
for i in ids:
    print(f'  {i:7d} -> {tok.decode([i])!r}')