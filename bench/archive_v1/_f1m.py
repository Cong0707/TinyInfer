GB = 2**30
W = 286.4
KV_PER_SEQ = 1.250
BW_CARD = 4.8
print("1M context, fp8 KV, weights resident 286 GB")
print()
for n, cap in ((4, 564.0), (8, 1128.0)):
    room = cap - W
    maxc = int(room / KV_PER_SEQ)
    bw = n * BW_CARD
    eff = bw * 0.7
    print(f"=== {n} x H200: {cap:.0f} GB, KV room {room:.0f} GB, max {maxc} seq @1M ===")
    print("  conc   KV GB    read GB   ms/tok   per-seq   total")
    for conc in (1, 4, 8, 16, 32, 64, 128, maxc):
        kv = KV_PER_SEQ * conc
        if kv > room or conc < 1:
            continue
        b = 11.0 + kv
        ms = b / eff * 1000
        per = 1000 / ms
        print(f"  {conc:5d} {kv:8.2f} {b:10.1f} {ms:8.1f} {per:9.1f} {per*conc:8.0f}")
    print()