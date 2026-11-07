"""Correct accounting: RAM is a cache, so only the misses come from disk."""
GB = 2**30
TOTAL = 47.6          # Qwen 27B bf16
RAM_USABLE = 30.0     # measured free on this box
VRAM = 8.0
DISK = 2.85           # GB/s measured
PCIE = 12.0           # GB/s host->device

print("Qwen3.8-27B bf16 = 47.6 GB")
print(f"usable RAM = {RAM_USABLE} GB, VRAM = {VRAM} GB")
print()
# If the engine pins as much as fits in RAM, every token only reads the remainder.
cached = min(TOTAL, RAM_USABLE)
miss = TOTAL - cached
print(f"resident in RAM   : {cached:5.1f} GB")
print(f"moved per token   : {miss:5.1f} GB  ({(miss/TOTAL)*100:.1f}% of the model)")
print(f"  from disk       : {miss:5.1f} GB / {DISK} GB/s = {miss/DISK:5.1f} s/token")
print(f"  disk->RAM->VRAM : {miss/DISK + miss/PCIE:5.1f} s/token "
      f"(adds {miss/PCIE:.1f}s of PCIe)")
print(f"  -> {1/(miss/DISK + miss/PCIE):.3f} tok/s")
print()
print("And with a smarter split: keep the layers that never change in VRAM too.")
for vram_use in (6.0, 7.0):
    ram_left = RAM_USABLE
    miss2 = TOTAL - vram_use - ram_left
    if miss2 < 0: miss2 = 0
    print(f"  VRAM {vram_use:.0f}GB + RAM {ram_left:.0f}GB -> miss {miss2:5.1f} GB/token"
          f" -> {1/(miss2/DISK + miss2/PCIE):.3f} tok/s" if miss2 > 0 else
          f"  VRAM {vram_use:.0f}GB + RAM {ram_left:.0f}GB -> fully resident")
print()
print("DeepSeek comparison: 5.59 GB/token, of which the disk part is 5.59 because the")
print("model is 510GB and RAM holds only ~30GB of it. That is why it needs streaming")
print("from disk every step, and why 1.93 tok/s is already near its floor.")
