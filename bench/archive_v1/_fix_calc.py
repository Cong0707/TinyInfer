GB = 2**30; MB = 2**20

print("=== what a token ACTUALLY reads (corrected) ===")

# 1. dense layers: attention projections etc. -- every layer, every token
dense = 5.39
print(f"  40 layers dense (MLA/norms/gate)  : {dense:6.2f} GB  <- unavoidable")

# 2. routed experts: 240 of 15360
experts = 4.31
print(f"  240 routed experts                : {experts:6.2f} GB")

# 3. head: ONE gemv of the full vocab projection = 1.26 GB read once
head = 1.2625
print(f"  head gemv (129280 x 5120)         : {head:6.2f} GB  <- read once per token")

# 4. embedding: ONE row lookup, ~20 KB
emb_row = 5120 * 2 / MB
print(f"  embedding row                     : {emb_row:6.4f} MB  <- negligible")

total = dense + experts + head
print(f"  {"TOTAL":34s}: {total:6.2f} GB/token")
print()
print("my earlier 12.93 GB double-counted the embedding table; the real figure is")
print(f"{total:.2f} GB/token")
print()

HBM, TBS = 141.0, 4.8
for n in (4, 8):
    agg = n * TBS
    ms = total / agg * 1000
    print(f"{n} x H200 ({agg:.1f} TB/s): {total:.2f} GB / {agg:.1f} TB/s = "
          f"{ms:.0f} ms -> {1000/ms:.1f} tok/s single stream")
print()
print("and the latency floor is not just bandwidth:")
print("  40 layers in a dependency chain, each with 2-4 kernels")
print("  ~150 kernel launches, ~5 us each = 0.75 ms of pure launch overhead")