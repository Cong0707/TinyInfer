import sys
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.budget import HostPool
p = HostPool.from_claimed(bytes(1024*1024), 1024*1024)
print('RESULT total:', p.total, 'free:', p.free)
v = p.alloc(512*1024)
print('RESULT alloc 512K:', type(v).__name__ if v else None)
v2 = p.alloc(512*1024)
print('RESULT alloc another 512K:', type(v2).__name__ if v2 else None)
v3 = p.alloc(1024)
print('RESULT alloc 1K (should be None):', v3)