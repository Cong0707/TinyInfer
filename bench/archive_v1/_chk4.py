import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.kernels.quant import act_quant
x = torch.randn(1, 512, dtype=torch.bfloat16, device='cuda')
q, s = act_quant(x, 32, 'ue8m0', torch.float8_e8m0fnu)
print(f'RESULT s dtype={s.dtype}')
print(f'RESULT s as float: {s.float().flatten()[:4].tolist()}')
print(f'RESULT s as uint8: {s.view(torch.uint8).flatten()[:4].tolist()}')
print()
print('RESULT 若 kernel 期望 fp32 = 具体值(0.0078)，传 .float() 正确')
print('RESULT 若 kernel 期望 e8m0 字节(127)，传 .float() 就错了')
print('RESULT 关键：gemv_fp4_pq 第 38 行做了 .float()')
print('RESULT 但注释说「e8m0 -> fp32 扩展」是 act_quant 之后才做的')
# 看 kernel 怎么用 asc
import subprocess
r = subprocess.run(['findstr','/C:asc','E:\\Develop\\Deepseek\\tinyinfer\\src\\tinyinfer\\kernels\\cuda\\gemv_fp4_w4a8.cu'], capture_output=True, text=True)
for line in (r.stdout or '').split(chr(10))[:8]: print('  ', line.strip()[:90])