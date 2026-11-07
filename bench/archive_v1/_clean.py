"""Remove the older kernels from the cu file, keeping only fast3 as the sole d4 user."""
from pathlib import Path
p = Path(r"E:\Develop\Deepseek\tinyinfer\src\tinyinfer\native\gemv_fp4_e8m0.cu")
src = p.read_text(encoding="utf-8")

# keep: the file header comment + the fp4_readonly diagnostic + the LAST fast3 block
import re
# find all kernel entry points
marks = [m.start() for m in re.finditer(r"extern \"C\" __global__ void (\w+)", src)]
names = re.findall(r"extern \"C\" __global__ void (\w+)", src)
print("kernels in file:", names)

# rebuild: header up to the first duplicate d4, then the readonly kernel, then fast3 only
head_end = src.find("extern \"C\" __global__ void")
head = src[:head_end]

def block(name):
    i = src.find(f"extern \"C\" __global__ void {name}")
    if i < 0:
        return ""
    # up to the next kernel decl or EOF
    j = src.find("extern \"C\" __global__ void", i + 10)
    return src[i:j if j > 0 else len(src)]

ro = block("fp4_readonly")
f3 = block("gemv_fp4_fast3")
# d4 helper needed by fast3
d4_start = src.find("__device__ __forceinline__ float d4(unsigned int c) {")
d4_end = src.find("\n}\n", d4_start) + 3
d4 = src[d4_start:d4_end]

out = head.rstrip() + "\n\n" + d4 + "\n" + ro + "\n" + f3
p.write_text(out, encoding="utf-8")
print("rewrote, bytes:", len(out))
