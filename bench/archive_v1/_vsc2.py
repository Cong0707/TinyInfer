import sys
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.loading import VramSlotCache
c = VramSlotCache(4, 1<<20, device='cuda:0')
print('RESULT public API:', [a for a in dir(c) if not a.startswith('_')])