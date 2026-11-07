"""Decode with the CUDA graph captured over the engine's layer loop."""
import sys, time
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
import torch
import tinyinfer.torch_compat  # noqa: F401
from tinyinfer import gdn_patch, qwen_norm
gdn_patch.install(); qwen_norm.install()
from tinyinfer.engine import Engine
from tinyinfer.hf_graph import GraphDecoder
from transformers.cache_utils import StaticCache


def build_cache(model, cap):
    """A static KV window the graph can write into (DynamicCache allocates per token)."""
    return StaticCache(config=model._load_hf_config(), max_cache_len=cap)
from transformers import AutoTokenizer


def make(model_dir, cap=256):
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    eng = Engine.from_hf(model_dir, device="cuda", allow_missing_layers=True)
    kv = build_cache(eng.adapter.model, cap)     # a static, capturable cache
    return eng, tok, kv


def decode_with_graph(eng, tok, kv, prompt, n=64):
    from transformers.cache_utils import DynamicCache
    model = eng.adapter.model

    # prefill with the ordinary path (the graph only covers single-token steps)
    ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")
    dyn = model.make_kv(ids.shape[1] + n + 8)
    gen = eng.adapter.model.step(eng, dyn, ids, torch.arange(0, 1, device="cuda"))
    print("prefill ok", flush=True)

    # capture one step
    ids_buf = torch.zeros(1, 1, dtype=torch.long, device="cuda")
    pos_buf = torch.zeros(1, dtype=torch.long, device="cuda")
    pos_buf.fill_(ids.shape[1])
    logits_buf = torch.empty(1, eng.cfg.vocab_size, dtype=torch.float32, device="cuda")
    dec = GraphDecoder.from_functions(
        ids, lambda i, p: eng.adapter.model.step(eng, kv, i, p),
        logits_buf, eng.cfg.vocab_size, capacity=n + 8)
    print(f"captured: {dec.using_graph if hasattr(dec,'using_graph') else '?'}", flush=True)
    return dec


if __name__ == "__main__":
    MD = sys.argv[1] if len(sys.argv) > 1 else r"E:\Models\Qwen3.5-0.8B"
    eng, tok, kv = make(MD)
    print(f"resident {eng.gpu_layers}/{eng.cfg.n_layers}", flush=True)
    dec = decode_with_graph(eng, tok, kv, "hi", 64)
