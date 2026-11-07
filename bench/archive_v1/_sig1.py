import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.kernels import native
import inspect
# 看这两个 kernel 的签名
for name in ('rms_norm', 'rms_norm_rope', 'hc_pre_rms_norm', 'head_gemv'):
    try:
        k = native.kernel(name, None) if False else None
    except Exception:
        pass
import subprocess
for f in ('rms_norm','rms_norm_rope','hc_pre_rms_norm','head_gemv','moe_act'):
    p = rf'E:\Develop\Deepseek\tinyinfer\src\tinyinfer\kernels\cuda\{f}.cu'
    import pathlib
    if not pathlib.Path(p).exists():
        print(f'RESULT {f}: no .cu'); continue
    txt = pathlib.Path(p).read_text(encoding='utf-8', errors='ignore')
    import re
    for m in re.finditer(r'extern\s+"C"\s+__global__\s+void\s+(\w+)\s*\(([^)]*)\)', txt, re.S):
        args = ' '.join(m.group(2).split())[:160]
        print(f'RESULT {f}::{m.group(1)}({args})')