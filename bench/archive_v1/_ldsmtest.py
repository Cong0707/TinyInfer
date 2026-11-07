import subprocess, os, sys
os.chdir(r"E:\Develop\Deepseek\deepseek")
vc = r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
r = subprocess.run(f'cmd /c "call "{vc}" >nul 2>&1 && nvcc -cubin -arch=sm_120 -o _ldsm.cubin _ldsm.cu"',
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
    src = torch.arange(256, dtype=torch.float16, device="cuda")
    out = torch.zeros(32*4, dtype=torch.float32, device="cuda")
    N.launch(N.kernel("_ldsm","test_ldsm"), (1,), (32,), [src, out])
    torch.cuda.synchronize()
    print("ldmatrix OK, first lane values (as float bits):", out[:8].tolist())
except Exception as e:
    print("ldmatrix failed:", type(e).__name__, str(e)[:200])
finally:
    N._DIR = orig
