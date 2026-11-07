import sys, torch, time
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.kernels import native
from tinyinfer.loading import ExpertStore
from tinyinfer.weights import ShardStore
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
store = ShardStore(d)
# 用共享专家（fp8）测 fp8_gemv_bs
t = store.get('layers.3.ffn.shared_experts.w1.weight') if store.has('layers.3.ffn.shared_experts.w1.weight') else None
print('RESULT shared w1 exists:', t is not None)
if t is None:
    names = [n for n in store.names() if 'shared' in n][:5]
    print('RESULT candidates:', names)
else:
    print('RESULT w1:', tuple(t.shape), t.dtype)