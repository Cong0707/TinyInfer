import subprocess, os, sys
os.chdir(r"E:\Develop\Deepseek\deepseek")
vc = r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
r = subprocess.run(f'cmd /c "call "{vc}" >nul 2>&1 && nvcc -cubin -arch=sm_120 -o _frag.cubin _frag.cu"',
                   capture_output=True, text=True, shell=True)
print("nvcc:", r.returncode, r.stderr[:400] if r.returncode else "ok")
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
import torch
import tinyinfer.torch_compat  # noqa
import tinyinfer.native as N
from pathlib import Path
orig = N._DIR
N._DIR = Path(r"E:\Develop\Deepseek\deepseek")
try:
    torch.manual_seed(0)
    A = torch.randint(-4, 5, (16, 16), device="cuda").half()
    B = torch.randint(-4, 5, (16, 8), device="cuda").half()
    C = torch.zeros(16, 8, device="cuda", dtype=torch.float32)
    N.launch(N.kernel("_frag","frag_probe"), (1,), (32,), [A, B, C, 16, 8, 16])
    torch.cuda.synchronize()
    ref = (A.float() @ B.float())
    d = (C - ref).abs()
    print(f"max diff: {d.max().item():.4f}  mean {d.mean().item():.4f}")
    print("C[0]:", C[0].tolist())
    print("ref[0]:", ref[0].tolist())
finally:
    N._DIR = orig
