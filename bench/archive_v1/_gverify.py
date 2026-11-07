"""Verify the captured graph produces the same tokens as eager."""
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
prompt = tok.apply_chat_template([{"role":"user","content":"1+1等于几？"}],
                                 tokenize=False, add_generation_prompt=True)
ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")
T = int(ids.shape[1])
cap = T + 96
kv = StaticCache(config=m._load_hf_config(), max_cache_len=cap)
pos_all = torch.arange(0, cap, device="cuda").view(1, 1, -1)

with torch.inference_mode():
    # prefill
    h0 = F.embedding(ids, eng.embed)
    rope0 = m.position_embeddings(h0, torch.arange(0, T, device="cuda"))
    tp0 = m.text_positions(torch.arange(0, T, device="cuda"))
    full_mask = m.attention_mask(h0, pos_all, cache=kv, layer_id=m.kinds.index("full_attention"))
    h = h0
    for i in range(m.n_layers):
        lay = eng.ensure_layer(i)
        mk = full_mask[:, :, :, :T] if m.kinds[i] == "full_attention" else None
        got = lay.forward(h, cache=kv, position=tp0, position_embeddings=rope0, attention_mask=mk)
        h = got[0] if isinstance(got, tuple) else got
    first = int(F.linear(m.final_norm(h[:, -1:]), eng.head)[0, -1].argmax())

ids_buf = torch.zeros(1, 1, dtype=torch.long, device="cuda")
pos_buf = torch.zeros(1, dtype=torch.long, device="cuda")
out_buf = torch.empty(1, eng.cfg.vocab_size, dtype=torch.float32, device="cuda")

def step():
    p = pos_buf.to(torch.long)
    hh = F.embedding(ids_buf, eng.embed)
    rope = m.position_embeddings(hh, p)
    tp = m.text_positions(p)
    for i in range(m.n_layers):
        lay = eng.ensure_layer(i)
        mk = full_mask if m.kinds[i] == "full_attention" else None
        got = lay.forward(hh, cache=kv, position=tp, position_embeddings=rope, attention_mask=mk)
        hh = got[0] if isinstance(got, tuple) else got
    out_buf.copy_(F.linear(m.final_norm(hh[:, -1:]), eng.head).reshape(-1).float())

# --- eager reference: fresh cache, generate 24 tokens
def run(kind, n=24):
    global kv
    kv = StaticCache(config=m._load_hf_config(), max_cache_len=cap)
    with torch.inference_mode():
        hh = F.embedding(ids, eng.embed)
        pa = torch.arange(0, T, device="cuda")
        full_mask = m.attention_mask(hh, pos_all, cache=kv, layer_id=m.kinds.index("full_attention"))
        r0 = m.position_embeddings(hh, pa)
        t0 = m.text_positions(pa)
        cur = hh
        for i in range(m.n_layers):
            lay = eng.ensure_layer(i)
            mk = full_mask[:, :, :, :T] if m.kinds[i] == "full_attention" else None
            g_ = lay.forward(cur, cache=kv, position=t0, position_embeddings=r0, attention_mask=mk)
            cur = g_[0] if isinstance(g_, tuple) else g_
        nxt = int(F.linear(m.final_norm(cur[:, -1:]), eng.head)[0, -1].argmax())
        out = [nxt]
        pos = T
        ids_buf.fill_(nxt); pos_buf.fill_(pos)
        if kind == "graph":
            gg = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gg):
                step()
            torch.cuda.synchronize()
        for _ in range(n - 1):
            if kind == "graph":
                gg.replay()
            else:
                step()
            torch.cuda.synchronize()
            tok_id = int(out_buf[0].argmax())
            out.append(tok_id)
            ids_buf.fill_(tok_id); pos_buf.add_(1)
    return out

e = run("eager")
g = run("graph")
print("eager:", e[:12], flush=True)
print("graph:", g[:12], flush=True)
print("identical:", e == g, flush=True)
