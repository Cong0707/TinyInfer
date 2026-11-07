"""Is gemv_fp4_fast numerically the E2M1 decode? Test against the exact table."""
import sys
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
import torch
import tinyinfer.torch_compat  # noqa
from tinyinfer import native

N, K = 64, 512
words = torch.randint(0, 2**31 - 1, (N, K//8), dtype=torch.int64)
w_i32 = words.to(torch.int32).to("cuda")
w_u8 = w_i32.view(torch.uint8)
ws_u8 = torch.full((N, K//32), 127, dtype=torch.uint8, device="cuda")
torch.manual_seed(0)
x = torch.randn(K, dtype=torch.bfloat16, device="cuda")
xf = x.float()
asc = torch.ones(K//32, dtype=torch.float32, device="cuda")
y = torch.zeros(N, dtype=torch.bfloat16, device="cuda")

u = words & 0xFFFFFFFF
codes = torch.zeros((N, K), dtype=torch.int64)
for i in range(8):
    codes[:, i::8] = (u >> (4 * i)) & 0xF
e2m1 = torch.tensor([0,0.5,1,1.5,2,3,4,6, 0,-0.5,-1,-1.5,-2,-3,-4,-6])
ref = e2m1[codes] @ xf.cpu()

native.launch(native.kernel("gemv_fp4_e8m0","gemv_fp4_fast"), ((N+7)//8,), (256,),
              [xf, asc, w_u8, ws_u8, y, N, K])
torch.cuda.synchronize()
d = (y.cpu().float() - ref).abs()
print(f"scalar fast vs E2M1 reference: max {d.max().item():.4f} rel {(d.max()/ref.abs().max()).item():.2e}")
print(f"  y  [0:4]: {[round(v,3) for v in y[:4].tolist()]}")
print(f"  ref[0:4]: {[round(v,3) for v in ref[:4].tolist()]}")
