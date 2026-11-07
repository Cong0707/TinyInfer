import sys, time
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
import torch
import tinyinfer.torch_compat  # noqa: F401
from tinyinfer import gdn_patch, qwen_norm
gdn_patch.install(); qwen_norm.install()
from tinyinfer.engine import Engine

t0 = time.perf_counter()
try:
    eng = Engine.from_hf(r"E:\Models\Qwen3.5-0.8B", device="cuda", allow_missing_layers=True)
    print(f"engine up in {time.perf_counter()-t0:.1f}s | family={eng.family} "
          f"| layers={eng.cfg.n_layers}", flush=True)
    print("embed:", tuple(eng.embed.shape), "head:", tuple(eng.head.shape), flush=True)
    b0 = eng.block(0)
    print("layer0:", type(b0).__name__, "kind:", getattr(b0, "kind", "?"),
          "device:", getattr(b0, "device", "?"), flush=True)
    print("weight paths on layer0:", len(b0._weight_paths()), flush=True)
except Exception:
    import traceback; traceback.print_exc()
