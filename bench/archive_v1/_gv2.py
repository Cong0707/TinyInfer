"""Correctness + speed of the captured step, in one pass."""
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
prompt = tok.apply_chat_template([{"role": "user", "content": "1+1等于几？"}],
                                 tokenize=False, add_generation_prompt=True)
ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")
T = int(ids.shape[1]); cap = T + 96
pos_all = torch.arange(0, cap, device="cuda").view(1, 1, -1)

ids_buf = torch.zeros(1, 1, dtype=torch.long, device="cuda")
pos_buf = torch.zeros(1, dtype=torch.long, device="cuda")
out_buf = torch.empty(1, eng.cfg.vocab_size, dtype=torch.float32, device="cuda")

def make_state():
    kv = StaticCache(config=m._load_hf_config(), max_cache_len=cap)
    h = F.embedding(ids, eng.embed)
    pa = torch.arange(0, T, device="cuda")
    mask = None      # the reference passes None and lets SDPA pick its causal path
    rope = m.position_embeddings(h, pa)
    tp = m.text_positions(pa)
    cur = h
    for i in range(m.n_layers):
        lay = eng.ensure_layer(i)
        mk = mask
        got = lay.forward(cur, cache=kv, position=tp, position_embeddings=rope, attention_mask=mk)
        cur = got[0] if isinstance(got, tuple) else got
    return kv, mask, int(F.linear(m.final_norm(cur[:, -1:]), eng.head)[0, -1].argmax())

def step_fn(kv, mask):
    def step():
        p = pos_buf.to(torch.long)
        hh = F.embedding(ids_buf, eng.embed)
        rope = m.position_embeddings(hh, p)
        tp = m.text_positions(p)
        for i in range(m.n_layers):
            lay = eng.ensure_layer(i)
            mk = mask if m.kinds[i] == "full_attention" else None
            got = lay.forward(hh, cache=kv, position=tp, position_embeddings=rope, attention_mask=mk)
            hh = got[0] if isinstance(got, tuple) else got
        out_buf.copy_(F.linear(m.final_norm(hh[:, -1:]), eng.head).reshape(-1).float())
    return step

def generate(kind, n=20):
    kv, mask, first = make_state()
    step = step_fn(kv, mask)
    out = [first]
    ids_buf.fill_(first); pos_buf.fill_(T)
    with torch.inference_mode():
        if kind == "graph":
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                step()
            torch.cuda.synchronize()
        for _ in range(n - 1):
            if kind == "graph":
                g.replay()
            else:
                step()
            torch.cuda.synchronize()
            t = int(out_buf[0].argmax())
            out.append(t); ids_buf.fill_(t); pos_buf.add_(1)
    return out

eager = generate("eager")
graph = generate("graph")
print("eager:", eager[:10], flush=True)
print("graph:", graph[:10], flush=True)
print("identical:", eager == graph, flush=True)
print("text:", repr(tok.decode(graph)), flush=True)
