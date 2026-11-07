"""Isolate why capture fails: is it the per-layer mask/rope allocation?"""
import sys
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
import torch
import tinyinfer.torch_compat  # noqa: F401
from tinyinfer import gdn_patch, qwen_norm
gdn_patch.install(); qwen_norm.install()
from tinyinfer.engine import Engine
from transformers import AutoTokenizer
import torch.nn.functional as F

MD = r"E:\Models\Qwen3.5-0.8B"
tok = AutoTokenizer.from_pretrained(MD, trust_remote_code=True)
eng = Engine.from_hf(MD, device="cuda", allow_missing_layers=True)
model = eng.adapter.model
ids = tok("hi", return_tensors="pt").input_ids.to("cuda")
T = int(ids.shape[1])

# Precompute everything the step needs, so the captured region allocates nothing:
# one rope table pair and one mask per layer kind, sized for the whole window.
from transformers.cache_utils import StaticCache
cap = T + 128
kv = StaticCache(config=model._load_hf_config(), max_cache_len=cap)
pos_all = torch.arange(0, cap, device="cuda")

with torch.inference_mode():
    h0 = F.embedding(ids, eng.embed)
    rope = model.position_embeddings(h0, pos_all[:T])
    print("rope shapes:", [tuple(t.shape) for t in rope], flush=True)
    masks = {}
    for kind in ("full_attention", "linear_attention"):
        idx = model.kinds.index(kind)
        masks[idx] = model.attention_mask(h0, pos_all[:T].view(1, -1), cache=kv, layer_id=idx)
    for i, mk in masks.items():
        print(f"mask[{i}]:", type(mk).__name__,
              [tuple(t.shape) for t in mk] if isinstance(mk, tuple) else getattr(mk, "shape", "?"),
              flush=True)
