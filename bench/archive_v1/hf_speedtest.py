import os, time, sys

os.environ.setdefault("HF_HOME", r"E:\Develop\Deepseek\deepseek\hf")
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

mode = sys.argv[1] if len(sys.argv) > 1 else "hub"
repo = "deepseek-ai/DeepSeek-V4.1-Flash-1017"

if mode == "mirror":
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import hf_hub_download

t0 = time.time()
p = hf_hub_download(repo_id=repo, filename="tokenizer.json", cache_dir=os.path.join(os.environ["HF_HOME"], "hub"))
dt = time.time() - t0
size = os.path.getsize(p)
print(f"mode={mode} size={size/1e6:.1f}MB time={dt:.2f}s speed={size/dt/1e6:.2f} MB/s")
