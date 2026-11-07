GB = 2**30; MB = 2**20
DENSE_L = 175.0 * MB
EXP1 = 18.38 * MB
L = 40; TOPK = 6
print(f"dense  {DENSE_L/MB:.0f} MB/layer, experts 384 x {EXP1/MB:.2f} MB")
per_dense = DENSE_L * L / GB
per_exp = TOPK * L * EXP1 / GB
per_head = 1.26
total = per_dense + per_exp + per_head
print(f"per token: dense {per_dense:.2f} + experts {per_exp:.2f} + head {per_head:.2f} = {total:.2f} GB")
print()
TBS = 4.8
for n in (1, 2, 3, 4):
    bw = n * TBS
    ms = total / bw * 1000
    print(f"{n} x H200 ({bw:.1f} TB/s): {ms:.1f} ms -> {1000/ms:.1f} tok/s single stream")
print()
print("batch scaling (weights shared, FLOPs multiply):")
ACTIVE = 37e9
for n in (2, 3):
    flops = n * 1979e12
    for b in (1, 8, 32):
        tps = flops / (2 * ACTIVE * b) * b
        print(f"  {n} cards batch {b:2d}: compute ceiling {flops/(2*ACTIVE):.0f} tok/s total")