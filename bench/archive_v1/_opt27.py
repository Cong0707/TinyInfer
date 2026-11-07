"""Quantified options for making 27B fast on 8GB VRAM + 30GB RAM."""
MB = 2**20; GB = 2**30
total = 47.6
# option A: quantise to 4-bit (what a modern engine does for this size)
for bits, label in ((4, "Q4"), (3, "Q3"), (2, "Q2")):
    size = total * bits / 16
    fits_ram = size <= 30
    print(f"{label}: {size:5.1f} GB -> {'FITS in 30GB RAM' if fits_ram else 'still streams'}")
print()
# if it fits in RAM, the cost per token becomes a RAM->VRAM copy of one layer at a time
for bits, label in ((4, "Q4"), (3, "Q3")):
    size = total * bits / 16
    per_layer = size / 64
    for bw, nm in ((12.0, "PCIe4 x8"), (25.0, "PCIe5 x16")):
        sec = per_layer / bw
        print(f"{label} layer {per_layer*1024:.0f}MB over {nm} ({bw} GB/s): "
              f"{sec*1000:5.1f} ms/layer -> 64 layers = {sec*64:4.2f}s/token")
print()
# option B: what fits in 8GB VRAM alone, and how fast that runs
print("VRAM-only (no streaming): 8GB holds", f"{8/total*16:.1f}-bit of the full model")
print("  -> a 2-bit 5.9GB build would fit, at obvious quality cost")
