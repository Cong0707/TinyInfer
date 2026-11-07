import sys, torch, shutil, pathlib
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
torch.cuda.init(); torch.zeros(1, device='cuda')  # 初始化 CUDA
from tinyinfer.kernels import native
# 备份原 cubin，换新的
build = pathlib.Path(r'E:\Develop\Deepseek\tinyinfer\src\tinyinfer\kernels\build')
orig = build / 'gemv_fp4_w4a8.cubin'
bak = build / 'gemv_fp4_w4a8.cubin.orig'
if not bak.exists(): shutil.copy2(orig, bak)
shutil.copy2(r'E:\temp\gemv_new.cubin', orig)
# 清 native 的模块缓存并重载
native._MODULES = {} if hasattr(native,'_MODULES') else None
print('RESULT attrs:', [a for a in dir(native) if 'mod' in a.lower() or 'cache' in a.lower()][:6])