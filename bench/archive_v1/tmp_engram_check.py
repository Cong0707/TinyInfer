import sys, traceback
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from pathlib import Path
from transformers import AutoTokenizer
from tinyinfer.config import TinyConfig
from tinyinfer.weights import ShardStore
from tinyinfer.engram import EngramHashes

md = Path(r"E:\Models\DeepSeek-V4.1-Flash-1017")
cfg = TinyConfig.from_hf(md)
store = ShardStore(md)
tok = AutoTokenizer.from_pretrained(str(md), trust_remote_code=True)
shards = {store.shard_of(f"layers.{i}.engram.embed.weight") for i in cfg.engram_layer_ids}
print("shards:", shards)
print("exists:", all((store.model_dir / s).exists() for s in shards))
try:
    h = EngramHashes(cfg, tok)
    print("hash ok:", h.n_cols)
except Exception:
    traceback.print_exc()
