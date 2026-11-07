import sys, torch, time
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
import tinyinfer.kernels.quant as Q
K = 5120
x = torch.randn(1, K, dtype=torch.bfloat16, device='cuda')
def bench(label, fn, n=2000):
    for _ in range(50): fn()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/n*1e6
print(f'RESULT full quantize_act : {bench("f", lambda: Q.act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu)):.2f} us')
z = x.contiguous()
print(f'RESULT x.contiguous()   : {bench("c", lambda: x.contiguous()):.2f} us')
print(f'RESULT view              : {bench("v", lambda: z.view(1, K)):.2f} us')
print(f'RESULT new_empty scale   : {bench("s", lambda: z.new_empty(1, K//32, dtype=torch.float8_e8m0fnu)):.2f} us')
print(f'RESULT torch.empty fp8   : {bench("e", lambda: torch.empty((1,K), dtype=torch.float8_e4m3fn, device=z.device)):.2f} us')
print(f'RESULT _native_ok()      : {bench("n", Q._native_ok):.2f} us')
s = z.new_empty(1, K//32, dtype=torch.float8_e8m0fnu)
y = torch.empty((1,K), dtype=torch.float8_e4m3fn, device=z.device)
print(f'RESULT _launch only      : {bench("l", lambda: Q._launch(z.view(1,K), y, s, "act_quant")):.2f} us')