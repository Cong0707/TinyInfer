"""Run the ACTUAL Marlin kernel from IST-DASLab on this GPU, same problem size as mine.

That is the only apples-to-apples comparison available offline: Marlin is what vLLM uses
for W4A16 fp4, it is Apache-2.0, and it is one .cu file plus a launcher.
"""
import urllib.request, os
base = "https://raw.githubusercontent.com/IST-DASLab/marlin/master/"
files = ["marlin/marlin_cuda_kernel.cu", "marlin/marlin_cuda.cpp", "marlin/kernel.h"]
os.makedirs(r"E:\Develop\Deepseek\deepseek\_marlin_src", exist_ok=True)
for f in files:
    try:
        req = urllib.request.Request(base + f, headers={"User-Agent": "Mozilla/5.0"})
        data = urllib.request.urlopen(req, timeout=30).read()
        out = os.path.join(r"E:\Develop\Deepseek\deepseek\_marlin_src", os.path.basename(f))
        open(out, "wb").write(data)
        print(f"{f}: {len(data)} bytes")
    except Exception as e:
        print(f"{f}: {e}")
