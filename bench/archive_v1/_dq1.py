import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.dequant_kernel import dequant_fp8
O, I = 256, 512
torch.manual_seed(0)
w = (torch.randn(O,I,device='cuda')*100).to(torch.float8_e4m3fn)
s = torch.randint(120,132,(O//32,I//32),dtype=torch.uint8,device='cuda')
out = dequant_fp8(w, s)
# 参考
scale_f = torch.pow(2.0, (s.to(torch.int32)-127).float())
ref = w.float() * scale_f.repeat_interleave(32,0).repeat_interleave(32,1)
d = (out.float()-ref).abs()
print(f'RESULT maxdiff={d.max().item():.4e} rel={d.max().item()/ref.abs().max().item():.2e}')
print(f'RESULT shape {tuple(out.shape)} dtype {out.dtype}')
# 性能
import time
for _ in range(10): dequant_fp8(w,s)
torch.cuda.synchronize(); t0=time.perf_counter()
for _ in range(50): dequant_fp8(w,s)
torch.cuda.synchronize(); dt=(time.perf_counter()-t0)/50
bytes_ = O*I + O*I//32*1
print(f'RESULT {dt*1e6:.1f} us, {bytes_/dt/1e9:.1f} GB/s')