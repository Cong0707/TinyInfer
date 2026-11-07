import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
import tinyinfer.kernels.native as N
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
mods = list(N._MODULES.keys()) if hasattr(N,'_MODULES') else []
print(f'RESULT loaded native modules: {len(mods)}')
for m in mods: print(f'RESULT   {m}')