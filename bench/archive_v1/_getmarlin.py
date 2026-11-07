import urllib.request, json
req = urllib.request.Request(
    "https://raw.githubusercontent.com/IST-DASLab/marlin/master/marlin/marlin_cuda_kernel.cu",
    headers={"User-Agent": "Mozilla/5.0"})
src = urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "replace")
open(r"E:\Develop\Deepseek\deepseek\_marlin_ref.cu", "w", encoding="utf-8").write(src)
print("saved", len(src), "bytes")
# print the key parts
i = src.find("cp_async_fence")
print(src[i:i+1200])
