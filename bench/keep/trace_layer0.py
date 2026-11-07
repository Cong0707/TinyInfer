
import sys, time, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(r"E:\Develop\Deepseek\tinyinfer")
sys.path.insert(0, str(ROOT / "src"))
from tinyinfer.config import TinyConfig
from tinyinfer.weights import ShardStore
from tinyinfer.model import Block
from tinyinfer.nn import rms_norm

torch.set_num_threads(16)
MODEL = ROOT / "cache" / "weights"
cfg = TinyConfig.from_hf(MODEL)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(str(MODEL), trust_remote_code=True)
text = """def quicksort(xs):
    if len(xs) <= 1:
        return xs
    pivot = xs[len(xs)//2]
    left = [x for x in xs if x < pivot]
    mid = [x for x in xs if x == pivot]
    right = [x for x in xs if x > pivot]
    return quicksort(left) + mid + quicksort(right)

The quick brown fox jumps over the lazy dog. DeepSeek V4.1 Flash uses a sparse mixture of experts with 384 routed experts per layer. Only six are activated per token, plus one shared expert. The router scores are sqrt(softplus(gate logits)) and the top six are normalized and scaled by 1.5."""
ids = tok(text).input_ids[:192]
print("prompt tokens:", len(ids))

with ShardStore(MODEL) as st:
    block = Block(st, 0, cfg)
    embed = st.get("embed.weight")
    h = embed[torch.tensor(ids, dtype=torch.long).unsqueeze(0)]
    h = h.unsqueeze(2).repeat(1, 1, cfg.hc_mult, 1)  # [1, s, hc, d]
    pre_mix = torch.zeros(1, len(ids), cfg.hc_mult)
    pre_mix[:, :, 0] = 1.0

    t0 = time.perf_counter()
    # replicate Block.forward up to the router, skipping the FFN experts
    residual = h
    attn_pre, attn_post, attn_comb = block.hc_mixes(h, block.hc_attn_fn, block.hc_attn_scale, block.hc_attn_base)
    x = block.hc_pre(h, pre_mix)
    x = rms_norm(x, block.attn_norm, cfg.rms_norm_eps)
    x = block.attn.forward(x, 0)
    x = block.hc_post(x, residual, attn_post, attn_comb)
    ffn_pre, ffn_post, ffn_comb = block.hc_mixes(x, block.hc_ffn_fn, block.hc_ffn_scale, block.hc_ffn_base)
    h2 = block.hc_pre(x, attn_pre)
    h2 = rms_norm(h2, block.ffn_norm, cfg.rms_norm_eps)
    gate = block.ffn.gate
    scores = F.linear(h2.float(), gate.weight)
    scores = F.softplus(scores).sqrt()
    indices = (scores + gate.bias).topk(cfg.n_activated_experts, dim=-1)[1]
    dt = time.perf_counter() - t0
    print(f"layer0 attn+router: {dt:.2f}s for {len(ids)} tokens")

routes = indices.reshape(-1, cfg.n_activated_experts).numpy()
n = len(routes)
print("routes shape", routes.shape)

# distribution stats
counts = np.bincount(routes.reshape(-1), minlength=cfg.n_routed_experts)
freq = np.sort(counts)[::-1]
share = freq / freq.sum()
print("unique experts:", int((counts > 0).sum()), "of", cfg.n_routed_experts)
for k in (6, 12, 24, 48, 96):
    print(f"  top-{k:3d} share: {share[:k].sum()*100:5.1f}%")
ent = -(share[share>0] * np.log(share[share>0])).sum()
print(f"  effective experts (exp(entropy)): {np.exp(ent):.1f}")

# adjacent-position overlap (temporal locality inside prefill)
ov = [len(set(routes[i]) & set(routes[i+1])) for i in range(n-1)]
print(f"adjacent position overlap: mean={np.mean(ov):.2f}/6  P(>=1)={np.mean(np.array(ov)>=1):.2f} "
      f"P(all 6)={np.mean(np.array(ov)==6):.3f}")

# per-layer LRU cache simulation over positions (layer 0 only)
class LRU:
    def __init__(self, cap): self.cap=cap; self.d={}
    def access(self, es):
        hit=0
        for e in es:
            if e in self.d:
                hit+=1; self.d.pop(e); self.d[e]=1
            else:
                if len(self.d)>=self.cap: self.d.pop(next(iter(self.d)))
                self.d[e]=1
        return hit
for cap in (12, 20, 27, 40, 64):
    lru = LRU(cap); hits=0
    for r in routes:
        hits += lru.access(r.tolist())
    print(f"  per-layer LRU cap={cap:2d}: hit={hits/(n*6)*100:5.1f}%")
np.savez(r"E:\Develop\Deepseek\deepseek\layer0_routes.npz", routes=routes, counts=counts)
print("saved layer0_routes.npz")
