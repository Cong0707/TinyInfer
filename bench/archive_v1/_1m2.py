GB = 2**30
print("=== 1M 上下文 decode: weights shared, KV per-sequence ===")
print()
W = 10.96   # GB of weights a token path reads, ONCE per batch
KV = 1.25   # GB per sequence at 1M ctx, fp8

for n, bw in ((4, 19.2), (8, 38.4)):
    print(f"=== {n} x H200, {bw:.1f} TB/s ===")
    print("  conc  bytes/step GB   ms/step   per-seq tok/s   total tok/s")
    for conc in (1, 4, 8, 16, 32, 64, 128, 222):
        b = W + KV * conc
        ms = b / bw * 1000
        per = 1000 / ms
        print(f"  {conc:5d} {b:12.1f} {ms:11.1f} {per:13.1f} {per*conc:12.0f}")
    print()

print("规律: small conc -> weights dominate; past ~9 seq the KV matches weights")
print("      beyond that, total is flat (bandwidth bound), per-seq drops")
print()
ACTIVE = 37e9
print("=== compute ceiling ===")
for n, fp8 in ((4, 4*1979e12), (8, 8*1979e12)):
    print(f"  {n} x H200 fp8: {fp8/(2*ACTIVE):.0f} tok/s")
print()
print("=== which binds at 1M context ===")
print("  4 cards: bandwidth 14305 seq-tok/s vs compute 106973 -> bandwidth")
print("  8 cards: bandwidth 28610 vs compute 213946 -> bandwidth")
print("  at 1M context H200 is KV-bandwidth bound, not compute bound")