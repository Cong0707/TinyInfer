import urllib.request, os
base = "https://raw.githubusercontent.com/IST-DASLab/marlin/master/marlin/"
for f in ("kernel.h", "marlin_cuda_kernel.cu", "dequant.h"):
    try:
        req = urllib.request.Request(base + f, headers={"User-Agent": "Mozilla/5.0"})
        data = urllib.request.urlopen(req, timeout=30).read()
        out = os.path.join(r"E:\Develop\Deepseek\deepseek\_marlin_src", f)
        open(out, "wb").write(data)
        print(f"{f}: {len(data)} bytes")
    except Exception as e:
        print(f"{f}: {type(e).__name__} {e}")
# list what the repo has
try:
    req = urllib.request.Request("https://api.github.com/repos/IST-DASLab/marlin/contents/marlin",
                                 headers={"User-Agent": "Mozilla/5.0"})
    import json
    items = json.loads(urllib.request.urlopen(req, timeout=30).read())
    print("repo files:", [i["name"] for i in items])
except Exception as e:
    print("listing failed:", e)
