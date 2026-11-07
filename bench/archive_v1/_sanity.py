GB = 2**30
# sanity: does 12.93 GB/token make sense for a 600B-class MoE?
W_RESIDENT = 286.4
PER_TOKEN = 12.93
print(f"resident weights      : {W_RESIDENT} GB")
print(f"reads per token       : {PER_TOKEN} GB ({100*PER_TOKEN/W_RESIDENT:.0f}% of them)")
print()
# dense part is unavoidable, experts are 240/15360 blobs
dense = 5.39; experts = 4.31; tables = 3.23
print(f"  dense 40 layers     : {dense} GB  <- every token")
print(f"  240 of 15360 experts: {experts} GB  ({240/15360*100:.1f}% of expert pool)")
print(f"  embed+head          : {tables} GB  <- every token")
print()
print("So per token the model touches 6.1% of its resident bytes.")
print(f"A 4-card H200 has {4*4.8} TB/s; 12.93 GB / 19.2 TB/s = {PER_TOKEN/(4*4.8)*1000:.0f} ms")
print(f"A 8-card H200 has {8*4.8} TB/s; 12.93 GB / 38.4 TB/s = {PER_TOKEN/(8*4.8)*1000:.0f} ms")
print()
print("for comparison, this same model on my laptop:")
print("  5.59 GB/token (measured) / 2.85 GB/s disk = 1961 ms -> 1.93 tok/s measured")
print("  the laptop is disk bound; H200 is HBM bound: 19.2 TB/s is 6700x the disk")