import sys
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.server import GenerationRequest, SamplingParams, mock_backend_render if False else None
from tinyinfer.chat import encode_messages
# 验证两种 thinking_mode 的差异（这是 server 会选的）
msgs = [{'role':'user','content':'hi'}]
a = encode_messages(msgs, thinking_mode='chat')
b = encode_messages(msgs, thinking_mode='thinking')
print('RESULT chat mode:', repr(a))
print('RESULT thinking mode:', repr(b))
print('RESULT differ:', a != b)
print('RESULT chat has </think>:', '</think>' in a)
print('RESULT thinking has <think>:', '<think>' in b and '</think>' not in b)