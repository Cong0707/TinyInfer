import subprocess, os, sys
os.chdir(r"E:\Develop\Deepseek\deepseek")
vc = r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
r = subprocess.run(f'cmd /c "call "{vc}" >nul 2>&1 && nvcc -cubin -arch=sm_120 -o _deqdump.cubin _deqdump.cu"',
                   capture_output=True, text=True, shell=True)
print("nvcc:", r.returncode, r.stderr[:300] if r.returncode else "ok")
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
import torch
import tinyinfer.torch_compat  # noqa
import tinyinfer.native as N
from pathlib import Path
orig = N._DIR
N._DIR = Path(r"E:\Develop\Deepseek\deepseek")
try:
    # one word = 8 nibbles. Build a word whose nibbles are 0,1,...,7 low->high
    word = 0
    for i in range(8):
        word |= (i & 0xF) << (4 * i)
    w = torch.tensor([word], dtype=torch.int32, device="cuda")
    out = torch.zeros(4, dtype=torch.float16, device="cuda")
    N.launch(N.kernel("_deqdump","deq_dump"), (1,), (1,), [w, out, 1])
    torch.cuda.synchronize()
    print("word nibbles 0..7 -> halves:", out.tolist())
    print("expected (signed nibble as int4): [0,1,2,3, 4,5,6,7]")
finally:
    N._DIR = orig
