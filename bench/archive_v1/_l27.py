import sys, os, time
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
import torch
import tinyinfer.torch_compat  # noqa: F401
from tinyinfer import gdn_patch, qwen_norm
gdn_patch.install(); qwen_norm.install()
from transformers import AutoModelForCausalLM, AutoTokenizer

MD = r"E:\Models\Qwen3.8-27B"
t0 = time.perf_counter()
tok = AutoTokenizer.from_pretrained(MD, trust_remote_code=True)
print(f"tokenizer: {time.perf_counter()-t0:.1f}s", flush=True)

t0 = time.perf_counter()
try:
    model = AutoModelForCausalLM.from_pretrained(
        MD, dtype=torch.bfloat16, trust_remote_code=True,
        device_map="auto", max_memory={0: "6GiB", "cpu": "28GiB"},
        low_cpu_mem_usage=True)
    print(f"loaded in {time.perf_counter()-t0:.1f}s", flush=True)
    devs = {}
    for n, p in model.named_parameters():
        d = str(p.device)
        devs[d] = devs.get(d, 0) + p.numel() * p.element_size()
    for d, b in devs.items():
        print(f"  {d}: {b/2**30:.2f} GB")
except Exception as exc:
    import traceback; traceback.print_exc()
    print("LOAD FAILED:", type(exc).__name__)
