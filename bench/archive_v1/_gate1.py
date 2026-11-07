import sys, torch, time
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.weights import ShardStore
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
st = ShardStore(d)
# gate 矩阵多大？
w = st.get('layers.3.ffn.gate.weight')
print('RESULT gate w', tuple(w.shape), w.dtype)
b = st.get('layers.3.ffn.gate.bias')
print('RESULT gate bias', tuple(b.shape))
n = w.shape[0]*w.shape[1] * (2 if w.dtype==torch.bfloat16 else 1)
print(f'RESULT gate bytes {n/2**20:.2f} MB')
print(f'RESULT 40 层 gate 合计 {40*n/2**20:.1f} MB —— 可以全部放 CPU 常驻')
# CPU 上算一次 gate 要多久？
import numpy as np
g32 = w.float().numpy() if hasattr(w,'numpy') else None
x = torch.randn(1, w.shape[1], dtype=torch.float32)
xf = x.numpy()
t0 = time.perf_counter()
for _ in range(20): _ = xf @ g32.T
t = (time.perf_counter()-t0)/20
print(f'RESULT CPU gate matmul {t*1000:.2f} ms/层 -> 40层 {40*t*1000:.0f} ms')