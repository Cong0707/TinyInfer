import sys, torch, time
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
torch.cuda.init(); torch.zeros(1, device='cuda')
def bench(fn,n=5000):
    for _ in range(50): fn()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/n*1e6
print(f'RESULT current_stream()        : {bench(lambda: torch.cuda.current_stream()):.3f} us')
c = torch.cuda.current_stream()
print(f'RESULT .cuda_stream attr       : {bench(lambda: c.cuda_stream):.3f} us')
print(f'RESULT int(.cuda_stream)       : {bench(lambda: int(c.cuda_stream)):.3f} us')
d = torch.cuda.current_device()
print(f'RESULT current_device()        : {bench(lambda: torch.cuda.current_device()):.3f} us')
print(f'RESULT ctypes.c_void_p(int)    : {bench(lambda: __import__("ctypes").c_void_p(int(c.cuda_stream))):.3f} us')