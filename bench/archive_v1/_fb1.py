import sys, torch, ctypes
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
torch.cuda.init(); torch.zeros(1, device='cuda')
from tinyinfer.kernels import native
# 直接把 fatbin 当 cubin 加载试试
import shutil, pathlib
b = pathlib.Path(r'E:\Develop\Deepseek\tinyinfer\src\tinyinfer\kernels\build\dequant_fp8.cubin')
if not b.exists(): print('RESULT no cubin yet')
fb = pathlib.Path(r'E:\temp\deq_multi.fatbin')
dst = b.with_suffix('.cubin.bak')
if b.exists() and not dst.exists(): shutil.copy2(b, dst)
shutil.copy2(fb, b)
native._MODULES.clear()
try:
    fn = native.kernel('dequant_fp8', 'dequant_fp8')
    print(f'RESULT fatbin loaded ok, fn={fn}')
except Exception as e:
    print(f'RESULT FATBIN FAILED: {repr(e)[:150]}')