import sys
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
import torch
import tinyinfer.torch_compat
from tinyinfer import native
N, K = 4096, 5120
w = torch.randint(0, 2**31-1, (N, K//8), dtype=torch.int32, device="cuda").view(torch.uint8)
ws = torch.full((N, K//32), 127, dtype=torch.uint8, device="cuda")
x = torch.randn(K, dtype=torch.float32, device="cuda")
asc = torch.ones(K//32, dtype=torch.float32, device="cuda")
y = torch.zeros(N, dtype=torch.bfloat16, device="cuda")
for _ in range(3):
    native.launch(native.kernel("gemv_fp4_e8m0","gemv_fp4_fast4"), ((N+7)//8,), (256,), [x, asc, w, ws, y, N, K])
torch.cuda.synchronize()
