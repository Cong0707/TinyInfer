import sys, torch, time, ctypes
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.kernels import native
from tinyinfer.deepseek import fp4_native
# 加载新 cubin（用不同模块名避免冲突）
import ctypes
cu = ctypes.CDLL('nvcuda.dll')
mod = ctypes.c_void_p()
cu.cuModuleLoad.restype = ctypes.c_int
rc = cu.cuModuleLoad(ctypes.byref(mod), r'E:\temp\gemv_new.cubin')
print('RESULT load rc =', rc)
fn = ctypes.c_void_p()
cu.cuModuleGetFunction.restype = ctypes.c_int
sym = b'gemv_fp4_w4a8\x00'
rc2 = cu.cuModuleGetFunction(ctypes.byref(fn), mod, sym)
print('RESULT getfunc rc =', rc2)
if rc2 != 0:
    sys.exit(1)