"""Capture only the parts that do not allocate: rope table for a fixed length + mask reuse."""
import sys
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
import torch
import tinyinfer.torch_compat  # noqa: F401
from tinyinfer import gdn_patch, qwen_norm
gdn_patch.install(); qwen_norm.install()
from tinyinfer.engine import Engine
from transformers import AutoTokenizer
from transformers.cache_utils import StaticCache
import torch.nn.functional as F

MD = r"E:\Models\Qwen3.5-0.8B"
tok = AutoTokenizer.from_pretrained(MD, trust_remote_code=True)
eng = Engine.from_hf(MD, device="cuda", allow_missing_layers=True)
m = eng.adapter.model
ids = tok("hi", return_tensors="pt").input_ids.to("cuda")
T = int(ids.shape[1])
cap = T + 128
kv = StaticCache(config=m._load_hf_config(), max_cache_len=cap)

# Precompute the causal mask for the full window once; the captured step slices it.
pos_all = torch.arange(0, cap, device="cuda").view(1, 1, -1)
with torch.inference_mode():
    h0 = F.embedding(ids, eng.embed)
    full_mask = m.attention_mask(h0, pos_all, cache=kv, layer_id=m.kinds.index("full_attention"))
print("mask shape:", tuple(full_mask.shape), flush=True)

ids_buf = torch.zeros(1, 1, dtype=torch.long, device="cuda")
pos_buf = torch.zeros(1, dtype=torch.long, device="cuda")
out_buf = torch.empty(1, eng.cfg.vocab_size, dtype=torch.float32, device="cuda")

def step_no_alloc():
    """Same as model.step but with precomputed rope and mask (allocation-free)."""
    p = pos_buf.to(torch.long)
    h = F.embedding(ids_buf, eng.embed)
    rope = m.position_embeddings(h, p)
    tp = m.text_positions(p)
    for i in range(m.n_layers):
        lay = eng.ensure_layer(i)
        mask = full_mask if m.kinds[i] == "full_attention" else None
        got = lay.forward(h, cache=kv, position=tp, position_embeddings=rope, attention_mask=mask)
        h = got[0] if isinstance(got, tuple) else got
    row = F.linear(m.final_norm(h[:, -1:]), eng.head)
    out_buf.copy_(row.reshape(-1).float())

with torch.inference_mode():
    pos_buf.fill_(T); ids_buf.fill_(int(ids[0, -1]))
    step_no_alloc(); torch.cuda.synchronize()
    print("eager ok", flush=True)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        step_no_alloc()
    torch.cuda.synchronize()
    print("captured ok", flush=True)
def bench(fn, n=50, warm=10):
    for _ in range(warm): fn()
    torch.cuda.synchronize()
    e0 = torch.cuda.Event(True); e1 = torch.cuda.Event(True)
    e0.record()
    for _ in range(n): fn()
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1) / n

with torch.inference_mode():
    ms_e = bench(step_no_alloc)
    ms_g = bench(lambda: g.replay())
print(f"eager : {ms_e:.2f} ms -> {1000/ms_e:.1f} tok/s", flush=True)
print(f"graph : {ms_g:.2f} ms -> {1000/ms_g:.1f} tok/s", flush=True)
