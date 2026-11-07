GB = 2**30; MB = 2**20
DENSE_L = 138 * MB
EXP_L = 384 * 18.38 * MB
EXP1 = 18.38 * MB
TABLES = 3.23 * GB
L = 40; TOPK = 6
print(f"dense {DENSE_L/MB:.0f} MB/layer, experts {EXP_L/GB:.2f} GB/layer")
print(f"total dense {DENSE_L*L/GB:.1f} GB, total experts {EXP_L*L/GB:.1f} GB")
print()
HBM = 141.0
PCIE = 64.0
print(f"H200: {HBM} GB HBM, PCIe 5.0 x16 = {PCIE} GB/s")
print()
for n in (1, 2, 3):
    vram = n * HBM
    dense_all = DENSE_L * L / GB
    room = vram - dense_all - TABLES
    total_ex = EXP_L * L / GB
    frac = max(0.0, min(1.0, room / total_ex))
    print(f"=== {n} x H200 = {vram:.0f} GB ===")
    print(f"  dense {dense_all:.1f} + tables {TABLES:.1f} -> expert room {room:.1f} GB")
    print(f"  expert cache: {frac*100:.0f}% of {total_ex:.1f} GB")
    per_tok = TOPK * L * EXP1 / GB
    miss = per_tok * (1 - frac)
    print(f"  per token: {TOPK*L} experts = {per_tok:.2f} GB, miss {miss:.2f} GB")
    print(f"  PCIe {PCIE} GB/s -> {miss/PCIE*1000:.0f} ms -> {PCIE/miss:.1f} tok/s")
    print()