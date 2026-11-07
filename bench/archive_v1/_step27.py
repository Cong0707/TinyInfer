"""27B single step through model.step, which owns the device handling."""
import sys, time
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
import torch, psutil
import tinyinfer.torch_compat  # noqa: F401
from tinyinfer import gdn_patch, qwen_norm
gdn_patch.install(); qwen_norm.install()
from tinyinfer.engine import Engine
from transformers import AutoTokenizer

MD = r"E:\Models\Qwen3.8-27B"
tok = AutoTokenizer.from_pretrained(MD, trust_remote_code=True)
eng = Engine.from_hf(MD, device="cuda", allow_missing_layers=True)
m = eng.adapter.model
print(f"plan {eng.hw_plan.gpu_layers}/{m.n_layers} | cache {eng.hw_plan.cache_bytes/2**30:.1f}GB "
      f"| holds {m._resident_layers}", flush=True)
# one resident layer, not two: 737MB of weights is ~2.4GB of VRAM with slack, and the
# streamed layer needs room beside it
eng.gpu_layers = 1
kv = m.make_kv(32)
ids = tok("hi", return_tensors="pt").input_ids.to("cuda")
pos = torch.tensor([0], device="cuda")
t0 = time.perf_counter()
try:
    with torch.inference_mode():
        logits = m.step(eng, kv, ids, pos)
    dt = time.perf_counter() - t0
    print(f"step ok in {dt:.1f}s | top {int(logits[0,-1].argmax())} | "
          f"RSS {psutil.Process().memory_info().rss/2**30:.1f}GB", flush=True)
    print(f"-> {1/dt:.3f} tok/s (first token, cold cache)", flush=True)
except Exception as e:
    import traceback; traceback.print_exc(limit=3)
    print("FAILED:", type(e).__name__, flush=True)
