import subprocess, os, sys
os.chdir(r"E:\Develop\Deepseek\deepseek")
vc = r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
subprocess.run(f'cmd /c "call "{vc}" >nul 2>&1 && nvcc -cubin -arch=sm_120 -o _deqbench.cubin _deqbench.cu"',
               capture_output=True, text=True, shell=True)

sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
import torch
import tinyinfer.torch_compat  # noqa
import tinyinfer.native as N
from pathlib import Path
orig = N._DIR
N._DIR = Path(r"E:\Develop\Deepseek\deepseek")
try:
    n = 4 * 1024 * 1024          # 4M ints = 16M weights
    q = torch.randint(0, 256, (n,), dtype=torch.int32, device="cuda")
    o1 = torch.zeros(n*4, dtype=torch.float16, device="cuda")
    o2 = torch.zeros(n*4, dtype=torch.float32, device="cuda")
    def bench(fn, k=50):
        with torch.inference_mode():
            for _ in range(10): fn()
            torch.cuda.synchronize()
            e0 = torch.cuda.Event(True); e1 = torch.cuda.Event(True)
            e0.record()
            for _ in range(k): fn()
            e1.record(); torch.cuda.synchronize()
        return e0.elapsed_time(e1)/k
    ms1 = bench(lambda: N.launch(N.kernel("_deqbench","deq_lop3"), ((n+255)//256,), (256,), [q, o1, n]))
    ms2 = bench(lambda: N.launch(N.kernel("_deqbench","deq_scalar"), ((n+255)//256,), (256,), [q, o2, n]))
    gb = n * 4
    print(f"lop3   : {ms1*1000:8.1f} us -> {gb/2**30/(ms1/1000):6.1f} G-decode/s")
    print(f"scalar : {ms2*1000:8.1f} us -> {gb/2**30/(ms2/1000):6.1f} G-decode/s  x{ms2/ms1:.2f}")
finally:
    N._DIR = orig
