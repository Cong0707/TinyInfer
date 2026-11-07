import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.deepseek import fp4_native
from tinyinfer.deepseek.gemv_fp4_marlin import gemv_fp4_fast
from tinyinfer.loading import ExpertStore
from tinyinfer.weights import ShardStore
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
store = ShardStore(d)
es = ExpertStore.from_store(store, cache_bytes=2<<30, device='cuda')
with es.experts(3, [3]) as recs:
    t = recs[0].tensors_on('cuda')
    w1, s1 = t['w1.weight'], t['w1.scale']
K = w1.shape[1] * 2
torch.manual_seed(0)
# M=1 走 marlin（prefill 的单行）；M=1 也走 native（decode）
x = torch.randn(1, K, dtype=torch.bfloat16, device='cuda')
aq, asc = fp4_native.quantize_act(x, K)
dec = fp4_native.gemv_fp4_pq(aq, asc, w1, s1).float().flatten()
pre = gemv_fp4_fast(x, w1, s1).float().flatten()
diff = (dec - pre).abs()
print(f'RESULT decode[:3] {dec[:3].tolist()}')
print(f'RESULT prefill[:3] {pre[:3].tolist()}')
print(f'RESULT maxabs={diff.max().item():.4e}  mean={diff.mean().item():.4e}')
# 相对：以 decode 的量级为基准
scale = dec.abs().mean().item()
print(f'RESULT mean|v|={scale:.4f}  relative={diff.max().item()/scale:.4e}')
# bf16 ulp 参考
import math
ulp = math.ldexp(1.0, int(math.floor(math.log2(scale))) - 8 + 1)
print(f'RESULT bf16 ulp at that scale ~ {ulp:.4e}  ratio={diff.max().item()/ulp:.1f} ulp')