"""Get vLLM's full csrc for the kernels we need to baseline against.

vLLM 0.33 splits CUDA sources out of the wheel; the canonical source is the repo. Our local
repositories/vllm is a partial checkout (python only), so fetch csrc from GitHub at the pinned
commit and keep it next to the repo as a reference tree.
"""
import json, os, urllib.request, base64
API = "https://api.github.com/repos/vllm-project/vllm"
REF = "1c87991"   # the commit repositories/vllm is at

out = r"E:\Develop\Deepseek\repositories\vllm-csrc"
os.makedirs(out, exist_ok=True)

def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return urllib.request.urlopen(req, timeout=60).read()

try:
    tree = json.loads(get(f"{API}/git/trees/{REF}?recursive=1"))
    files = [t["path"] for t in tree.get("tree", [])
             if t["path"].startswith("csrc/") and t["path"].endswith((".cu", ".cuh", ".h", ".cpp"))]
    print(f"csrc files at {REF}: {len(files)}")
    for f in files[:15]:
        print("  ", f)
except Exception as e:
    print("tree failed:", e)
    files = []
print()
print("total:", len(files))
open(os.path.join(out, "_filelist.txt"), "w").write("\n".join(files))
