GB = 2**30; MB = 2**20
DENSE_L = 138 * MB
EXP1 = 18.38 * MB
L = 40; TOPK = 6
TABLES_GB = 3.23
DENSE_ALL_GB = DENSE_L * L / GB
EXP_ALL_GB = 384 * EXP1 * L / GB
print(f"dense {DENSE_ALL_GB:.1f} GB total, experts {EXP_ALL_GB:.1f} GB total")
print()
HBM = 141.0
PCIE = 64.0
RAM = 1536.0  # an H200 node has ~1.5-2 TB host RAM
print(f"H200: {HBM} GB HBM, PCIe {PCIE} GB/s, node RAM ~{RAM:.0f} GB")
print()
for n in (1, 2, 3, 4):
    vram = n * HBM
    room = vram - DENSE_ALL_GB - TABLES_GB
    frac = max(0.0, min(1.0, room / EXP_ALL_GB))
    print(f"=== {n} x H200 = {vram:.0f} GB ===")
    print(f"  dense {DENSE_ALL_GB:.1f} + tables {TABLES_GB:.1f} -> room {room:.1f} GB")
    print(f"  expert cache in VRAM: {frac*100:.0f}% of {EXP_ALL_GB:.1f} GB")
    per_tok = TOPK * L * EXP1 / GB
    # the rest comes from host RAM over PCIe
    miss = per_tok * (1 - frac)
    print(f"  per token {TOPK*L} experts = {per_tok:.2f} GB")
    print(f"    VRAM hit {frac*100:.0f}% -> PCIe miss {miss:.2f} GB")
    if miss > 0:
        print(f"    {miss/PCIE*1000:.0f} ms -> {PCIE/miss:.1f} tok/s")
    else:
        print(f"    all resident -> HBM bound")
    print()