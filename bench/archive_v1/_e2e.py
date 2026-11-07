"""End-to-end: engine + captured graph, measured over a bounded window."""
import sys, time
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
prompt = tok.apply_chat_template([{"role":"user","content":"hi"}],
                                 tokenize=False, add_generation_prompt=True)
ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")
T = int(ids.shape[1]); N = 64; cap = T + N + 1
kv = StaticCache(config=m._load_hf_config(), max_cache_len=cap)
kv.capacity = cap

def prefill():
    pa = torch.arange(0, T, device="cuda")
    h = F.embedding(ids, eng.embed)
    r0 = m.position_embeddings(h, pa); t0 = m.text_positions(pa)
    for i in range(m.n_layers):
        lay = eng.ensure_layer(i)
        got = lay.forward(h, cache=kv, position=t0, position_embeddings=r0, attention_mask=None)
        h = got[0] if isinstance(got, tuple) else got
    return int(F.linear(m.final_norm(h[:,-1:]), eng.head)[0,-1].argmax())

with torch.inference_mode():
    first = prefill()
    g, ids_buf, pos_buf, logits_buf = m.capture_step(eng, kv, ids, T)
    ids_buf.fill_(first); pos_buf.fill_(T)
    # warm up and check the window is not exceeded
    def one(t):
        g.replay()
        tok_id = int(logits_buf[0].argmax())
        ids_buf.fill_(tok_id); pos_buf.fill_(t)
        return tok_id
    for t in range(T, T + 8): one(t)
    torch.cuda.synchronize()
    # timed window
    t0 = time.perf_counter()
    for t in range(T + 8, T + 8 + N):
        one(t)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
print(f"graph decode: {N} tok in {dt:.3f}s -> {N/dt:.1f} tok/s", flush=True)

with torch.inference_mode():
    t0 = time.perf_counter()
    for t in range(T, T + N):
        m.step(eng, kv, ids_buf, torch.tensor([t], device="cuda"))
    torch.cuda.synchronize()
    dte = time.perf_counter() - t0
print(f"eager decode: {N} tok in {dte:.3f}s -> {N/dte:.1f} tok/s", flush=True)
