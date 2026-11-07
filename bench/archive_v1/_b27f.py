"""27B through the engine with the fixed plan: how fast, and how much comes from disk."""
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
tok = AutoTokenizer.from_pretrained(MD, trust_remote_code=True)
eng = Engine.from_hf(MD, device="cuda", allow_missing_layers=True)
m = eng.adapter.model
per = m.per_layer_bytes()
print(f"plan: {eng.hw_plan.gpu_layers}/{m.n_layers} resident | "
      f"cache {eng.hw_plan.cache_bytes/2**30:.1f} GB | "
      f"holds {m._resident_layers} layers", flush=True)
kv = m.make_kv(32)
ids = tok("hi", return_tensors="pt").input_ids.to("cuda")

def one_pass(tag):
    from collections import Counter
    hits = Counter()
    t0 = time.perf_counter()
    with torch.inference_mode():
        pos = torch.arange(0, ids.shape[1], device="cuda")
        h = F.embedding(ids.to(eng.embed.device), eng.embed).to("cuda")
        rope = m.position_embeddings(h, pos)
        tp = m.text_positions(pos)
        for i in range(m.n_layers):
            lay = eng.ensure_layer(i)
            before = getattr(lay, "_loaded", False)
            hits["cached" if before else "disk"] += 1
            dev = getattr(lay, "device", "cpu")
            work, pos_i = (h, tp) if dev == "cuda" else (h.cpu(), None)
            rope_i = rope if dev == "cuda" else tuple(t.cpu() for t in rope)
            if pos_i is None:
                pos_i = tp.cpu()
            out = lay.forward(work, cache=kv, position=pos_i,
                              position_embeddings=rope_i, attention_mask=None)
            h = out[0] if isinstance(out, tuple) else out
            if dev != "cuda":
                h = h.to("cuda")
            eng.release_layer(i) if False else None
        dt = time.perf_counter() - t0
        logits = F.linear(m.final_norm(h[:, -1:]), eng.head)
    print(f"{tag}: {dt:.1f}s | cached {hits['cached']} disk {hits['disk']} | "
          f"top {int(logits[0,-1].argmax())} | RSS {psutil.Process().memory_info().rss/2**30:.1f}GB",
          flush=True)
    return dt

d1 = one_pass("pass 1 (cold)")
d2 = one_pass("pass 2 (warm)")
print(f"-> cold {1/d1:.3f} tok/s | warm {1/d2:.3f} tok/s", flush=True)
