import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.weights import ShardStore
from tinyinfer.config import TinyConfig
from pathlib import Path

M = Path(r"E:\Develop\Deepseek\tinyinfer\cache\weights")
cfg = TinyConfig.from_hf(M)
print("cfg:", cfg.summary())
with ShardStore(M) as st:
    print("index tensors:", len(st.weight_map))
    emb = st.get("embed.weight")
    print("embed:", tuple(emb.shape), emb.dtype)
    f = emb.float()
    print(f"stats mean={f.mean():.4f} std={f.std():.4f}")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(M), trust_remote_code=True)
    ids = tok("Hello, I am DeepSeek.").input_ids
    e = emb[torch.tensor(ids)]
    print("ids:", ids)
    print("embedded:", tuple(e.shape), e.dtype, "norm:", float(e.float().norm()))
