import sys, torch, inspect
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.server import TinyModelBackend
print('RESULT sig:', str(inspect.signature(TinyModelBackend.__init__))[:280])