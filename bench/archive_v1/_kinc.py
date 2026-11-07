import sys
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
import torch
import tinyinfer.torch_compat  # noqa
from tinyinfer import native

def check(N, K):
    w = torch.zeros((N, K//8), dtype=torch.int32, device="cuda")
    w[:, :] = 0x11111111          # every code 1 -> decoded -7
    ws = torch.full((N, K//32), 127, dtype=torch.uint8, device="cuda")
    x = torch.ones(K, dtype=torch.bfloat16, device="cuda")
    y = torch.zeros(N, dtype=torch.bfloat16, device="cuda")
    native.launch(native.kernel("marlin_fp4","marlin_gemv2"), ((N + 63)//64,), (128,),
                  [x, w, ws, y, N, K])
    torch.cuda.synchronize()
    expect = -7.0 * K
    got = y[0].item()
    ok = abs(got - expect) < 0.5
    print(f"N={N:4d} K={K:4d}: got {got:10.2f}, expect {expect:10.2f}  {'OK' if ok else 'WRONG'}")
    return ok

for K in (32, 64, 96, 128, 160, 512):
    check(64, K)