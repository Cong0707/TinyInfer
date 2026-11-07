import sys
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from transformers import AutoTokenizer
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
msgs = [{'role':'user','content':'The capital of France is'}]
print('RESULT has template:', getattr(tok, 'chat_template', None) is not None)
try:
    out = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    print('RESULT template output:', repr(out))
except Exception as e:
    print('RESULT template FAILED:', repr(e)[:150])
# 我们的 chat.py
from tinyinfer.chat import render
import inspect
print('RESULT chat.py funcs:', [n for n,_ in inspect.getmembers(sys.modules['tinyinfer.chat'], inspect.isfunction)][:8])