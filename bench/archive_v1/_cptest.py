import subprocess, os, sys
os.chdir(r"E:\Develop\Deepseek\deepseek")
vc = r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
r = subprocess.run(f'cmd /c "call "{vc}" >nul 2>&1 && nvcc -cubin -arch=sm_120 -o _cptest.cubin _cptest.cu"',
                   capture_output=True, text=True, shell=True)
print("nvcc:", r.returncode, r.stderr[:500] if r.returncode else "ok")
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
import torch
import tinyinfer.torch_compat  # noqa
import tinyinfer.native as N
from pathlib import Path
orig = N._DIR
N._DIR = Path(r"E:\Develop\Deepseek\deepseek")
try:
    src = torch.arange(1024, dtype=torch.float16, device="cuda")
    for k in ("test_cpasync", "test_cpasync_policy"):
        out = torch.zeros(1024, dtype=torch.float16, device="cuda")
        try:
            N.launch(N.kernel("_cptest", k), (1,), (128,), [src, out, 1024])
            torch.cuda.synchronize()
            ok = torch.equal(out, src)
            print(f"{k}: OK, matches={ok}")
        except Exception as e:
            print(f"{k}: FAILED {type(e).__name__} {str(e)[:150]}")
finally:
    N._DIR = orig
