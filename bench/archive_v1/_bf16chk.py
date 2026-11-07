
import sys, struct
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")

# 独立复刻 marlin_dtypes.cuh 的 SM75 实现，和 torch 的 bf16 对比
def f2bf_sw(x: float) -> int:
    bits = struct.unpack("<I", struct.pack("<f", x))[0]
    lsb = (bits >> 16) & 1
    bits = (bits + 0x7FFF + lsb) & 0xFFFFFFFF
    return (bits >> 16) & 0xFFFF

def bf2f_sw(b: int) -> float:
    return struct.unpack("<f", struct.pack("<I", b << 16))[0]

import torch
torch.manual_seed(0)
vals = torch.randn(4096, dtype=torch.float32).tolist()
bad = 0
for v in vals:
    ref = torch.tensor(v).to(torch.bfloat16).view(torch.int16).item() & 0xFFFF
    got = f2bf_sw(v)
    if ref != got:
        bad += 1
        if bad <= 3:
            print(f"  不一致: {v:.6f} ref={ref:#06x} got={got:#06x}")
print(f"f32->bf16 往返: {len(vals)-bad}/{len(vals)} 一致")

# 反方向
bs = [f2bf_sw(v) for v in vals[:512]]
bad2 = sum(1 for b in bs if abs(bf2f_sw(b) - torch.tensor(b, dtype=torch.int16).view(torch.bfloat16).float().item()) > 0)
print(f"bf16->f32: {len(bs)-bad2}/{len(bs)} 一致")

# 特殊值
for name, v in (("0.0", 0.0), ("-0.0", -0.0), ("1.0", 1.0), ("-1.0", -1.0),
                ("inf", float("inf")), ("-inf", float("-inf"))):
    ref = torch.tensor(v).to(torch.bfloat16).view(torch.int16).item() & 0xFFFF
    got = f2bf_sw(v)
    ok = "OK" if ref == got else f"FAIL ref={ref:#06x} got={got:#06x}"
    print(f"  {name}: {ok}")
