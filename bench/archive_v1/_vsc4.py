import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.loading import VramSlotCache
import inspect
blob = 1 << 20
c = VramSlotCache(4, blob, device='cuda:0')
print('RESULT acquire sig:', inspect.signature(c.acquire))
print('RESULT copy_in sig:', inspect.signature(c.copy_in))
print('RESULT publish sig:', inspect.signature(c.publish))
print('RESULT release sig:', inspect.signature(c.release))
print('RESULT acquire_many sig:', inspect.signature(c.acquire_many))