import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
b = eng.block(4)
import inspect
src = inspect.getsource(b.__class__.forward)
# 打印前 60 行
for ln in src.splitlines()[:48]:
    print('  ', ln[:110])