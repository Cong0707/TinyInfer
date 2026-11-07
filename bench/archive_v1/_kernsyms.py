"""Measure the vLLM kernels that did compile. This is the baseline the rest is judged against."""
import sys, ctypes
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
import torch
import tinyinfer.torch_compat  # noqa
import tinyinfer.native as N
from pathlib import Path

orig = N._DIR
N._DIR = Path(r"E:\Develop\Deepseek\deepseek\_vlkern")
try:
    import subprocess, os
    CU = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin\cuobjdump.exe"
    for name in ("layernorm_kernels", "activation_kernels", "cache_kernels", "moe_align_sum_kernels"):
        p = Path(r"E:\Develop\Deepseek\deepseek\_vlkern") / f"{name}.cubin"
        if not p.exists():
            continue
        r = subprocess.run([CU, "-symbols", str(p)], capture_output=True, text=True)
        syms = [l for l in r.stdout.splitlines() if "STO_ENTRY" in l and "_vl_anchor" not in l
                and "EmptyKernel" not in l]
        print(f"{name}: {len(syms)} real kernels")
        for s in syms[:4]:
            # extract the mangled name (last token)
            print("   ", s.split()[-1][:90])
finally:
    N._DIR = orig
