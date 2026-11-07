import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
# 直接测：pin 一块，看 data_ptr 是否在合理范围
t = torch.empty(64 << 20, dtype=torch.uint8, pin_memory=True)
print('RESULT pinned data_ptr:', int(t.data_ptr()), hex(int(t.data_ptr())))
print('RESULT is_pinned:', t.is_pinned())
# frombuffer 同一个 numpy 数组
import numpy as np
a = np.empty(64 << 20, dtype=np.uint8)
print('RESULT numpy addr:', a.ctypes.data, hex(a.ctypes.data))
t2 = torch.from_numpy(a)
print('RESULT from_numpy data_ptr:', int(t2.data_ptr()))
# cudaHostRegister 呢
print('RESULT same?', int(t2.data_ptr()) == a.ctypes.data)
# torch 的 pin 是不是真的 cudaHostAlloc
print('RESULT t storage ptr:', int(t.untyped_storage().data_ptr()))