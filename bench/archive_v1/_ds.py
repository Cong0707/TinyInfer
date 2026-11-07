import subprocess, os, sys
os.chdir(r"E:\Develop\Deepseek\deepseek")
vc = r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
cmd = 'cmd /c "call "' + vc + '" >nul 2>&1 && nvcc -cubin -arch=sm_120 -o _dumpstage.cubin _dumpstage.cu"'
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
    Nn, K = 64, 32
    word = sum((i & 0xF) << (4*i) for i in range(8))
    w = torch.full((Nn, K//8), word, dtype=torch.int32, device="cuda")
    x = torch.arange(32, dtype=torch.bfloat16, device="cuda")
    outW = torch.zeros((16, 32), dtype=torch.float16, device="cuda")
    outA = torch.zeros(32, dtype=torch.float16, device="cuda")
    N.launch(N.kernel("_dumpstage","dump_stage"), ((Nn+63)//64,), (128,), [x, w, outW, outA, Nn, K])
    torch.cuda.synchronize()
    print("sA:", outA[:8].tolist(), "... expect 0,1,2,3,4,5,6,7")
    print("sW row0 k=0..7 :", outW[0][:8].tolist(), "... expect -8..-1")
    print("sW row0 k=8..15:", outW[0][8:16].tolist())
    print("sW row0 k=16..23:", outW[0][16:24].tolist())
finally:
    N._DIR = orig