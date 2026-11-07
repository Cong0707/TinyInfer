import urllib.request, os
out = r"E:\Develop\Deepseek\deepseek\_marlin_src"
for name in ("marlin.cu", "marlin.cuh", "kernel.h", "dequant.h", "marlin_dtypes.cuh", "marlin_mma.h", "marlin_template.h"):
    u = f"https://raw.githubusercontent.com/vllm-project/vllm/main/csrc/quantization/marlin/{name}"
    try:
        req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"})
        data = urllib.request.urlopen(req, timeout=30).read()
        open(os.path.join(out, name), "wb").write(data)
        print(f"{name}: {len(data)}")
    except Exception as e:
        print(f"{name}: {type(e).__name__}")
print()
print("files:", sorted(os.listdir(out)))
