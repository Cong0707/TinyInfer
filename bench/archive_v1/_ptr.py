
import ctypes
import torch

# 1) plain ctypes array
buf = (ctypes.c_char * 4096)()
mv = memoryview(buf)
t = torch.frombuffer(mv, dtype=torch.uint8)
print("plain   addressof:", ctypes.addressof(buf), " data_ptr:", t.data_ptr(),
      " diff:", t.data_ptr() - ctypes.addressof(buf))

# 2) pinned cudaHostAlloc style: allocate through torch then see
p = torch.empty(4096, dtype=torch.uint8, pin_memory=True)
print("pinned  data_ptr:", p.data_ptr())

# 3) what does torch.frombuffer over a memoryview of an untyped storage give
print("untyped storage ptr:", p.untyped_storage().data_ptr())
