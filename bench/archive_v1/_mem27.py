import sys, time, os
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
e = Engine.from_hf(MD, device="cuda", allow_missing_layers=True)
e.adapter.model._cache_cap = 2           # two layers at a time, not sixty-four
kv = e.adapter.make_kv(16)
m = e.adapter.model
ids = tok("hi", return_tensors="pt").input_ids.to("cuda")
print(f"before: RSS {proc.memory_info().rss/2**30:.2f} GB | "
      f"gpu {torch.cuda.memory_allocated()/2**30:.2f} GB", flush=True)
t0 = time.perf_counter()
peak = 0.0
with torch.inference_mode():
    pos = torch.arange(0, ids.shape[1], device="cuda")
    h = F.embedding(ids.to(e.embed.device), e.embed).to("cuda")
    rope = m.position_embeddings(h, pos)
    tp = m.text_positions(pos)
    for i in range(e.cfg.n_layers):
        lay = m.build_layer(i)
        if lay.device != "cuda":
            lay.to("cuda")
        mask = m.attention_mask(h, tp, cache=kv, layer_id=i)
        out = lay.forward(h, cache=kv, position=tp, position_embeddings=rope,
                          attention_mask=mask)
        h = out[0] if isinstance(out, tuple) else out
        if lay.device != "cpu":
            lay.to("cpu")
        if i % 16 == 0:
            rss = proc.memory_info().rss / 2**30
            peak = max(peak, rss)
            print(f"  layer {i}: RSS {rss:.2f} GB | cached {len(m._layers)}", flush=True)
    total = time.perf_counter() - t0
    logits = F.linear(m.final_norm(h[:, -1:]), e.head)
print(f"all {e.cfg.n_layers} layers in {total:.1f}s | peak RSS {peak:.2f} GB", flush=True)
print(f"top token: {int(logits[0,-1].argmax())}", flush=True)
