import sys, torch, time
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.deepseek import fp4_native as FN
for K in (5120, 2304, 1280):
    x = torch.randn(1, K, dtype=torch.bfloat16, device='cuda')
    for _ in range(20): FN.quantize_act(x, K)
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(500): FN.quantize_act(x, K)
    torch.cuda.synchronize(); dt=(time.perf_counter()-t0)/500
    print(f'RESULT quantize_act K={K}: {dt*1e6:.2f} us')
# 对比：一次 launch 的纯 CPU 开销
k = 5120
x = torch.randn(1, k, dtype=torch.bfloat16, device='cuda')
torch.cuda.synchronize(); t0=time.perf_counter()
for _ in range(500): FN.quantize_act(x, k)
torch.cuda.synchronize(); wall=(time.perf_counter()-t0)/500
# 纯 GPU 时间（异步提交，不等）
torch.cuda.synchronize(); t0=time.perf_counter()
for _ in range(500): FN.quantize_act(x, k)
submit=(time.perf_counter()-t0)/500
torch.cuda.synchronize()
print(f'RESULT wall={wall*1e6:.2f} us  submit_only={submit*1e6:.2f} us  gpu={(wall-submit)*1e6:.2f} us')