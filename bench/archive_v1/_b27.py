import sys, time, os
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
import torch
import tinyinfer.torch_compat  # noqa: F401
from tinyinfer.qwen_model import QwenModel

MD = r"E:\Models\Qwen3.8-27B"
t0 = time.perf_counter()
m = QwenModel.open(MD)
print(f"skeleton in {time.perf_counter()-t0:.1f}s | layers {m.n_layers}", flush=True)
print(f"RSS after skeleton: {__import__('psutil').Process().memory_info().rss/2**30:.2f} GB", flush=True)

# materialize ONE layer of the 27B: this is the whole claim -- 51.8GB checkpoint,
# one layer at a time, never the whole model
t0 = time.perf_counter()
lay = m.build_layer(0)
lay.materialize()
import psutil
print(f"layer0 materialized in {(time.perf_counter()-t0)*1000:.0f}ms", flush=True)
print(f"RSS now: {psutil.Process().memory_info().rss/2**30:.2f} GB", flush=True)
n = sum(p.numel() * p.element_size() for p in lay.module.parameters())
print(f"layer0 weights: {n/2**20:.1f} MB | device {next(lay.module.parameters()).device}",
      flush=True)
