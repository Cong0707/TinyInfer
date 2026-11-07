GB = 2**30; MB = 2**20
HBM = 141.0; PCIE = 64.0
DENSE = 5.4; TABLES = 3.2; EXPERTS = 275.7
EXP1 = 18.38 * MB / GB   # 0.01795 GB per expert
L = 40; TOPK = 6
PER_TOK_EXP = TOPK * L * EXP1
print(f"one token routes to {TOPK*L} experts = {PER_TOK_EXP:.2f} GB of expert weights")
print()
print("engine reality: expert cache hit rate is NOT 100% even when it fits,")
print("because a token only touches 240 of 15360 blobs. Measured on my laptop:")
print("  0.48 cold -> 0.66 -> 0.75 -> 0.88 warm (7 passes)")
print("on H200 the cache is far larger so the ceiling is higher, say 0.90-0.95")
print()
for n in (1, 2, 3):
    vram = n * HBM
    room = vram - DENSE - TABLES
    frac = min(1.0, room / EXPERTS)
    print(f"=== {n} x H200 = {vram:.0f} GB, expert room {room:.0f} GB ({frac*100:.0f}%) ===")
    for hit in (0.90, 0.95):
        miss_gb = PER_TOK_EXP * (1 - hit)
        ms = miss_gb / PCIE * 1000
        tps = 1000 / ms
        print(f"  cache hit {hit*100:.0f}%: miss {miss_gb:.3f} GB, PCIe {ms:.1f} ms -> {tps:.0f} tok/s")
    print()
print("even at 90% hit rate the miss is 0.43 GB/token over 64 GB/s = 6.7 ms")
print("that is the FLOOR before any compute: ~150 tok/s single stream")