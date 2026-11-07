"""Run vLLM's own Marlin kernel vs mine, same GPU, same problem shape.

Marlin expects weights in its own packed layout (k-major, interleaved), which vLLM prepares
with gptq_marlin_repack. To compare KERNEL throughput I do not need correct packing: I only
need both kernels to read the same number of bytes. So I feed Marlin a buffer of the right
size in the layout it expects and measure time.
"""
import sys, ctypes
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
import torch
import tinyinfer.torch_compat  # noqa
import tinyinfer.native as N
from pathlib import Path

orig = N._DIR
N._DIR = Path(r"E:\Develop\Deepseek\deepseek\_marlin_src")
try:
    # load the marlin cubin
    mod = N.module("marlin_bench")
    fn = N.function(mod, "marlin_test_launch")

    def bench(fncall, n=20, warm=5):
        with torch.inference_mode():
            for _ in range(warm): fncall()
            torch.cuda.synchronize()
            e0 = torch.cuda.Event(True); e1 = torch.cuda.Event(True)
            e0.record()
            for _ in range(n): fncall()
            e1.record(); torch.cuda.synchronize()
        return e0.elapsed_time(e1)/n

    print("marlin loaded:", fn is not None)
finally:
    N._DIR = orig
