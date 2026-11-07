import sys
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.chat import encode_messages
msgs = [{'role':'user','content':'hi'}]
a = encode_messages(msgs, thinking_mode='chat')
b = encode_messages(msgs, thinking_mode='thinking')
print('RESULT chat mode:', repr(a))
print('RESULT thinking mode:', repr(b))
print('RESULT differ:', a != b)
print('RESULT chat ends with /think:', a.rstrip().endswith('</think>'))
print('RESULT thinking ends with think:', b.rstrip().endswith('<think>'))