import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.server import TinyModelBackend
import inspect
src = inspect.getsource(TinyModelBackend)
# 找 load / forward 的绑定
for line in src.splitlines():
    if 'self._model' in line and ('=' in line or 'forward' in line):
        print('  ', line.strip()[:110])