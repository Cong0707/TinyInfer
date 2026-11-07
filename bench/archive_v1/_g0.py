"""0.8B through the engine, with and without a captured graph."""
import sys, time
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
import torch
import tinyinfer.torch_compat  # noqa: F401
from tinyinfer import gdn_patch, qwen_norm
gdn_patch.install(); qwen_norm.install()
from tinyinfer.engine import Engine
from transformers import AutoTokenizer
from transformers.cache_utils import StaticCache

MD = r"E:\Models\Qwen3.5-0.8B"
tok = AutoTokenizer.from_pretrained(MD, trust_remote_code=True)
eng = Engine.from_hf(MD, device="cuda", allow_missing_layers=True)
model = eng.adapter.model
ids = tok("hi", return_tensors="pt").input_ids.to("cuda")
T = int(ids.shape[1])
N = 64

def bench(fn, n=40, warm=10):
    for _ in range(warn := warm): fn()
    torch.cuda.synchronize()
    e0 = torch.cuda.Event(True); e1 = torch.cuda.Event(True)
    e0.record()
    for _ in range(n): fn()
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1) / n

# --- eager path (what we have) ---
kv = StaticCache(config=model._load_hf_config(), max_cache_len=T + N + 8)
with torch.inference_mode():
    model.step(eng, kv, ids, torch.tensor([0], device="cuda"))
    pos = torch.tensor([T], device="cuda")
    nxt = torch.zeros(1, 1, dtype=torch.long, device="cuda")
    def eager_step():
        lg = model.step(eng, kv, nxt, pos)
        nxt.fill_(int(lg[0, -1].argmax()))
        pos.add_(1)
    ms_eager = bench(eager_step)
print(f"eager  : {ms_eager:.2f} ms -> {1000/ms_eager:.1f} tok/s", flush=True)

# --- captured graph ---
with torch.inference_mode():
    model.step(eng, kv, ids, torch.tensor([0], device="cuda"))
    pos.fill_(T)
    nxt.zero_()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        lg = model.step(eng, kv, nxt, pos)
    torch.cuda.synchronize()
    def graph_step():
        g.replay()
    ms_g = bench(graph_step)
print(f"graph  : {ms_g:.2f} ms -> {1000/ms_g:.1f} tok/s", flush=True)
