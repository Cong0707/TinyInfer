import subprocess, os, sys
os.chdir(r"E:\Develop\Deepseek\deepseek")
vc = r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
cmd = 'cmd /c "call "' + vc + '" >nul 2>&1 && nvcc -cubin -arch=sm_120 -o _d8.cubin _d8.cu"'
r = subprocess.run(cmd, capture_output=True, text=True, shell=True)
print("nvcc:", r.returncode, r.stderr[:300] if r.returncode else "ok")
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
import torch
import tinyinfer.torch_compat  # noqa
import tinyinfer.native as N
from pathlib import Path
orig = N._DIR
N._DIR = Path(r"E:\Develop\Deepseek\deepseek")
try:
    # nibbles 0..7 in one word
    word = 0
    for i in range(8):
        word |= (i & 0xF) << (4 * i)
    w = torch.tensor([word], dtype=torch.int32, device="cuda")
    out = torch.zeros(8, dtype=torch.float16, device="cuda")
    N.launch(N.kernel("_d8","dump_deq8"), (1,), (1,), [w, out, 1])
    torch.cuda.synchronize()
    print("nibbles 0..7 ->", out.tolist())
    print("expected in k order: [-8,-7,-6,-5,-4,-3,-2,-1]")
finally:
    N._DIR = orig