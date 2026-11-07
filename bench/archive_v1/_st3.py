import subprocess, os, sys
os.chdir(r"E:\Develop\Deepseek\deepseek")
vc = r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
cmd = 'cmd /c "call "' + vc + '" >nul 2>&1 && nvcc -cubin -arch=sm_120 -o _stage3.cubin _stage3.cu"'
r = subprocess.run(cmd, capture_output=True, text=True, shell=True)
print("nvcc:", r.returncode, r.stderr[:400] if r.returncode else "ok")
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
import torch
import tinyinfer.torch_compat  # noqa
import tinyinfer.native as N
from pathlib import Path
orig = N._DIR
N._DIR = Path(r"E:\Develop\Deepseek\deepseek")
try:
    Nn, K = 64, 32
    w = torch.zeros((Nn, K//8), dtype=torch.int32, device="cuda")
    w[:, :] = 0x11111111          # every code = 1 -> decoded -7
    x = torch.ones(32, dtype=torch.bfloat16, device="cuda")
    out = torch.zeros((Nn, 8), dtype=torch.float32, device="cuda")
    N.launch(N.kernel("_stage3","stage3"), ((Nn+63)//64,), (128,), [x, w, out, Nn, K])
    torch.cuda.synchronize()
    print("stage3 OK")
    print("row0:", out[0].tolist())
    print("expect col0 = 16 * (-7) = -112 (one mma covers k=0..15)")
finally:
    N._DIR = orig