import sys, torch, time
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.dequant_kernel import dequant_fp8
# 用真实形状：2304x5120（专家 w1）
O, I = 2304, 5120
w = (torch.randn(O,I,device='cuda')*50).to(torch.float8_e4m3fn)
s = torch.randint(120,132,(O//32,I//32),dtype=torch.uint8,device='cuda')
out = dequant_fp8(w,s)
bytes_ = O*I + O*I//32 + O*I*2  # 读 fp8 + 读 scale + 写 bf16
for _ in range(5): dequant_fp8(w,s)
torch.cuda.synchronize(); t0=time.perf_counter()
for _ in range(20): dequant_fp8(w,s)
torch.cuda.synchronize(); dt=(time.perf_counter()-t0)/20
print(f'RESULT [{O},{I}] {dt*1e6:.1f} us  {bytes_/dt/1e9:.1f} GB/s')
# 参考 torch 路径
def torch_ver():
    sc = torch.pow(2.0,(s.to(torch.int32)-127).float()).repeat_interleave(32,0).repeat_interleave(32,1)
    return (w.float()*sc).to(torch.bfloat16)
for _ in range(3): torch_ver()
torch.cuda.synchronize(); t0=time.perf_counter()
for _ in range(20): torch_ver()
torch.cuda.synchronize(); dt2=(time.perf_counter()-t0)/20
print(f'RESULT torch  {dt2*1e6:.1f} us  speedup {dt2/dt:.1f}x')