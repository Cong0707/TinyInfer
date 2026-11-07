
import sys, struct, torch
torch.manual_seed(0)

def f2bf_sw(x: float) -> int:
    bits = struct.unpack("<I", struct.pack("<f", x))[0]
    lsb = (bits >> 16) & 1
    bits = (bits + 0x7FFF + lsb) & 0xFFFFFFFF
    return (bits >> 16) & 0xFFFF

def bf2f_sw(b: int) -> float:
    return struct.unpack("<f", struct.pack("<I", (b & 0xFFFF) << 16))[0]

vals = torch.randn(512, dtype=torch.float32)
refs = vals.to(torch.bfloat16)
bad = 0
for i, v in enumerate(vals.tolist()):
    got = f2bf_sw(v)
    ref = refs[i].view(torch.uint16).item()
    if got != ref:
        bad += 1
print(f"f32->bf16: {512-bad}/512  bit-exact")

# 反方向：用 numpy 相同位模式比较
import numpy as np
ok2 = 0
for i in range(512):
    b = refs[i].view(torch.uint16).item()
    got = bf2f_sw(b)
    ref = float(refs[i])
    if got == ref:
        ok2 += 1
print(f"bf16->f32: {ok2}/512  bit-exact")

for name, v in (("0.0",0.0),("-0.0",-0.0),("1.0",1.0),("-1.0",-1.0),
                ("inf",float("inf")),("-inf",float("-inf"))):
    ref = torch.tensor(v).to(torch.bfloat16).view(torch.uint16).item()
    got = f2bf_sw(v)
    print(f"  {name:5s}: {'OK' if ref==got else 'FAIL'}")
