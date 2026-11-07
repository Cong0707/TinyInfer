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
T = int(ids.shape[1]); cap = T + 128
kv = StaticCache(config=m._load_hf_config(), max_cache_len=cap)
ids_buf = torch.zeros(1,1,dtype=torch.long,device="cuda")
pos_buf = torch.zeros(1,dtype=torch.long,device="cuda")
out_buf = torch.empty(1, eng.cfg.vocab_size, dtype=torch.float32, device="cuda")

def step():
    p = pos_buf.to(torch.long)
    h = F.embedding(ids_buf, eng.embed)
    rope = m.position_embeddings(h, p)
    tp = m.text_positions(p)
    for i in range(m.n_layers):
        lay = eng.ensure_layer(i)
        got = lay.forward(h, cache=kv, position=tp, position_embeddings=rope,
                          attention_mask=None)
        h = got[0] if isinstance(got, tuple) else got
    out_buf.copy_(F.linear(m.final_norm(h[:,-1:]), eng.head).reshape(-1).float())

with torch.inference_mode():
    # prefill
    pa = torch.arange(0, T, device="cuda")
    h = F.embedding(ids, eng.embed)
    r0 = m.position_embeddings(h, pa); t0 = m.text_positions(pa)
    for i in range(m.n_layers):
        lay = eng.ensure_layer(i)
        got = lay.forward(h, cache=kv, position=t0, position_embeddings=r0, attention_mask=None)
        h = got[0] if isinstance(got, tuple) else got
    nxt = int(F.linear(m.final_norm(h[:,-1:]), eng.head)[0,-1].argmax())
    ids_buf.fill_(nxt); pos_buf.fill_(T)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        step()
    torch.cuda.synchronize()

def bench(fn, n=100, warm=20):
    with torch.inference_mode():
        for _ in range(warm): fn()
        torch.cuda.synchronize()
        e0 = torch.cuda.Event(True); e1 = torch.cuda.Event(True)
        e0.record()
        for _ in range(n): fn()
        e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1)/n

ms_e = bench(step); ms_g = bench(g.replay)
print(f"engine eager : {ms_e:.2f} ms -> {1000/ms_e:.1f} tok/s", flush=True)
print(f"engine graph : {ms_g:.2f} ms -> {1000/ms_g:.1f} tok/s", flush=True)
