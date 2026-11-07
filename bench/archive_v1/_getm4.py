import urllib.request, json, os
# find how vLLM builds marlin: it vendors kernel.h generated from marlin_template.h
cands = [
  "https://raw.githubusercontent.com/vllm-project/vllm/main/csrc/quantization/marlin/kernel.h",
  "https://raw.githubusercontent.com/vllm-project/vllm/main/csrc/quantization/marlin/dequant.h",
  "https://raw.githubusercontent.com/vllm-project/vllm/main/csrc/quantization/marlin/marlin.cuh",
  "https://raw.githubusercontent.com/vllm-project/vllm/main/csrc/quantization/marlin/marlin_dtypes.cuh",
  "https://raw.githubusercontent.com/vllm-project/vllm/main/csrc/quantization/marlin/marlin_mma.h",
  "https://raw.githubusercontent.com/vllm-project/vllm/main/csrc/quantization/marlin/marlin_template.h",
]
out_dir = r"E:\Develop\Deepseek\deepseek\_marlin_src"
for u in cands:
    name = u.rsplit("/", 1)[-1]
    try:
        req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"})
        data = urllib.request.urlopen(req, timeout=30).read()
        open(os.path.join(out_dir, name), "wb").write(data)
        print(f"{name}: {len(data)}")
    except Exception as e:
        print(f"{name}: {type(e).__name__}")
# also list vllm csrc marlin dir
try:
    req = urllib.request.Request("https://api.github.com/repos/vllm-project/vllm/contents/csrc/quantization/marlin",
                                 headers={"User-Agent": "Mozilla/5.0"})
    items = json.loads(urllib.request.urlopen(req, timeout=30).read())
    print("vllm marlin files:", [i["name"] for i in items])
except Exception as e:
    print("list failed:", e)
