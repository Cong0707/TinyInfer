
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
d = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
ids = tok("Hello", return_tensors="pt").input_ids
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device="cuda", layers=list(range(40)))
print("gpu_layers =", eng.gpu_layers, " max_layers =", eng.max_layers,
      " cuda_graph tier =", getattr(eng.tier, "cuda_graph", None))
print("_graph_capture_shape ->", eng._graph_capture_shape(ids, 1))
print("_graph_failed =", getattr(eng, "_graph_failed", "unset"))
