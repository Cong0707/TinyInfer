import subprocess, ctypes, os
os.chdir(r"E:\Develop\Deepseek\deepseek")
vc = r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
cmd = f'call "{vc}" >nul 2>&1 && nvcc -cubin -arch=sm_120 -o _deqchk.cubin _deqchk.cu'
r = subprocess.run(f'cmd /c "{cmd}"', capture_output=True, text=True)
print("nvcc rc:", r.returncode, r.stderr[:300] if r.returncode else "ok")

# load with the driver API through the project's helper
import sys
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
import torch
import tinyinfer.torch_compat  # noqa
from tinyinfer import native
import tinyinfer.native as N

# temporarily point module() at the probe cubin
orig = N._DIR
N._DIR = __import__("pathlib").Path(r"E:\Develop\Deepseek\deepseek")
try:
    q = torch.tensor([0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7,
                      0x8, 0x9, 0xA, 0xB, 0xC, 0xD, 0xE, 0xF], dtype=torch.int32, device="cuda")
    # pack two codes per int: use each as low nibble of a half2 pair
    n = q.numel()
    out = torch.zeros(n*2*2, dtype=torch.float16, device="cuda")
    N.launch(N.kernel("_deqchk", "check_decode"), ((n+31)//32,), (32,), [q, out, n])
    torch.cuda.synchronize()
    o = out.view(n, 2, 2).float()
    print("code -> decoded halves")
    for i in range(16):
        print(f"  0x{i:X} -> {o[i,0,0].item():8.3f}, {o[i,0,1].item():8.3f}, "
              f"{o[i,1,0].item():8.3f}, {o[i,1,1].item():8.3f}")
finally:
    N._DIR = orig
