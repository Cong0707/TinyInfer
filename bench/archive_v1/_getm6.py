import urllib.request, os
out = r"E:\Develop\Deepseek\deepseek\_marlin_src"
os.makedirs(os.path.join(out, "core"), exist_ok=True)
urls = {
  "core/scalar_type.hpp": "https://raw.githubusercontent.com/vllm-project/vllm/main/csrc/core/scalar_type.hpp",
  "torch_utils.h": "https://raw.githubusercontent.com/vllm-project/vllm/main/csrc/torch_utils.h",
}
for rel, u in urls.items():
    try:
        req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"})
        data = urllib.request.urlopen(req, timeout=30).read()
        open(os.path.join(out, rel), "wb").write(data)
        print(f"{rel}: {len(data)}")
    except Exception as e:
        print(f"{rel}: {type(e).__name__}")
