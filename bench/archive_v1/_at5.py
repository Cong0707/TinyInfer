import sys, torch, inspect
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.deepseek.attention import MLAWindowAttention
print('RESULT __init__:', str(inspect.signature(MLAWindowAttention.__init__))[:150])
print('RESULT forward:', str(inspect.signature(MLAWindowAttention.forward))[:150])
print('RESULT public:', [a for a in dir(MLAWindowAttention) if not a.startswith('_')][:14])