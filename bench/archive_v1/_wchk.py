import sys, time
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
import torch
import tinyinfer.torch_compat  # noqa: F401
from tinyinfer import gdn_patch, qwen_norm
gdn_patch.install(); qwen_norm.install()
from tinyinfer.qwen_model import QwenModel

MD = r"E:\Models\Qwen3.5-0.8B"
t0 = time.perf_counter()
m = QwenModel.open(MD)
lay0 = m.build_layer(0)
lay0.materialize()
print(f"layer0 materialized in {(time.perf_counter()-t0)*1000:.0f}ms", flush=True)

# compare against the same layer inside a normally-loaded model
from transformers import AutoModelForCausalLM
ref = AutoModelForCausalLM.from_pretrained(MD, dtype=torch.bfloat16,
                                           trust_remote_code=True).eval()
r0 = (ref.model.language_model.layers[0] if hasattr(ref.model, "language_model")
      else ref.model.layers[0])
diffs = []
mine = dict(lay0.module.named_parameters())
theirs = dict(r0.named_parameters())
worst = []
for k in mine:
    if k in theirs:
        a, b = mine[k], theirs[k]
        if a.shape == b.shape:
            d = float((a.float() - b.float()).abs().max())
            diffs.append(d)
            worst.append((d, k, tuple(a.shape), a.dtype, b.dtype))
worst.sort(reverse=True)
print("params compared:", len(diffs), "| max abs diff:", max(diffs) if diffs else "n/a", flush=True)
for d, k, shp, da, db in worst[:5]:
    print(f"  {d:.6f} {k} {shp} {da} vs {db}", flush=True)
