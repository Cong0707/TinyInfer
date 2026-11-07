GB = 2**30; MB = 2**20
HBM = 141.0; PCIE = 64.0; TBS = 4.8
DENSE = 5.4; TABLES = 3.2; EXPERTS = 275.7
EXP1 = 18.38 / 1024  # GB per expert
L = 40; TOPK = 6
PER_TOK = TOPK * L * EXP1
print(f"per token: {TOPK*L} experts = {PER_TOK:.2f} GB")
print()
print("=== 按 VRAM 容量算真实命中率 ===")
print()
print("H200 专家池 275.7 GB, 每个专家 18.38 MB, 40 层各 384 个")
print()
for n in (1, 2, 3):
    vram = n * HBM
    room = vram - DENSE - TABLES
    slots = int(room / (18.38/1024))
    frac = min(1.0, room / EXPERTS)
    print(f"{n} x H200 = {vram:.0f} GB, room {room:.0f} GB = {slots} expert slots ({frac*100:.0f}%)")
    # LRU hit rate ~ frac for uniformly random access; routing is skewed so better
    for skew, label in ((1.0, "uniform"), (1.3, "skewed")):
        hit = min(0.995, frac ** (1/skew))
        miss = PER_TOK * (1 - hit)
        pcie_ms = miss / PCIE * 1000 if miss > 0 else 0
        # total decode time: max(HBM path, PCIe path) roughly additive for misses
        hbm_gb = DENSE + TABLES + PER_TOK
        hbm_ms = hbm_gb / (n * TBS) * 1000
        total_ms = max(hbm_ms, pcie_ms)
        print(f"    {label}: hit {hit*100:.1f}%, miss {miss:.2f} GB, "
              f"PCIe {pcie_ms:.0f} ms, HBM {hbm_ms:.0f} ms -> {1000/total_ms:.0f} tok/s")
    print()