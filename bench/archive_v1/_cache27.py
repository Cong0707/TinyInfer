"""27B through the engine with the RAM cache active: hit rate and speed."""
import sys, time
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
import torch, psutil
import tinyinfer.torch_compat  # noqa: F401
from tinyinfer import gdn_patch, qwen_norm
gdn_patch.install(); qwen_norm.install()
from tinyinfer.engine import Engine
from transformers import AutoTokenizer
import torch.nn.functional as F

MD = r"E:\Models\Qwen3.8-27B"
proc = psutil.Process()
tok = AutoTokenizer.from_pretrained(MD, trust_remote_code=True)
eng = Engine.from_hf(MD, device="cuda", allow_missing_layers=True)
m = eng.adapter.model
print(f"cache budget GB: {eng.hw_plan.cache_bytes/2**30:.1f} | "
      f"per layer MB: {m.per_layer_bytes()/2**20:.0f} | "
      f"resident layers: {m._resident_layers}", flush=True)
kv = m.make_kv(32)
ids = tok("hi", return_tensors="pt").input_ids.to("cuda")

t0 = time.perf_counter()
with torch.inference_mode():
    pos = torch.arange(0, ids.shape[1], device="cuda")
    h = F.embedding(ids.to(eng.embed.device), eng.embed).to("cuda")
    rope = m.position_embeddings(h, pos)
    tp = m.text_positions(pos)
    for i in range(m.n_layers):
        lay = eng.ensure_layer(i)
        if lay.device != "cuda":
            lay.to("cuda")
        out = lay.forward(h, cache=kv, position=tp, position_embeddings=rope,
                          attention_mask=None)
        h = out[0] if isinstance(out, tuple) else out
        if i % 16 == 0:
            print(f"  layer {i}: {time.perf_counter()-t0:5.1f}s "
                  f"RSS {proc.memory_info().rss/2**30:.1f}GB", flush=True)
    total = time.perf_counter() - t0
    logits = F.linear(m.final_norm(h[:, -1:]), eng.head)
print(f"pass 1: {total:.1f}s | top {int(logits[0,-1].argmax())}", flush=True)
print(f"cached layers now: {len(m._layers)}", flush=True)

# second pass: everything resident should be a cache hit
t0 = time.perf_counter()
with torch.inference_mode():
    h = F.embedding(ids.to(eng.embed.device), eng.embed).to("cuda")
    for i in range(m.n_layers):
        lay = eng.ensure_layer(i)
        if lay.device != "cuda":
            lay.to("cuda")
        out = lay.forward(h, cache=kv, position=tp, position_embeddings=rope,
                          attention_mask=None)
        h = out[0] if isinstance(out, tuple) else out
    total2 = time.perf_counter() - t0
print(f"pass 2: {total2:.1f}s", flush=True)
