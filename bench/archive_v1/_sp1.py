import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.weights import ShardStore
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
st = ShardStore(d)
# 看一个专家的实际权重分布
w = st.get('layers.3.ffn.experts.0.w1.weight')
print('RESULT shape', tuple(w.shape), w.dtype)
# fp4 拆字节看
b = w.view(torch.uint8).reshape(-1)
lo = (b & 0x0F).long(); hi = ((b >> 4) & 0x0F).long()
import collections
c = collections.Counter(torch.cat([lo, hi]).tolist())
tot = sum(c.values())
print('RESULT fp4 code 分布:')
for k in sorted(c):
    print(f'  code {k:2d}: {100*c[k]/tot:5.2f}%')
zero = c.get(0,0) + c.get(8,0)
print(f'RESULT 零占比 {100*zero/tot:.2f}%')