import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.attention import MLAWindowAttention
import inspect
sig = inspect.signature(MLAWindowAttention.__init__)
print('RESULT __init__:', sig)
print('RESULT class attrs:', [a for a in dir(MLAWindowAttention) if not a.startswith('_')][:12])