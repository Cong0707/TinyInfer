"""Run the kernel once under ncu and dump the limiter."""
import sys, subprocess, os
bench_py = os.path.join(r"E:\Develop\Deepseek\deepseek", "_profrun.py")
open(bench_py, "w").write('''
import sys
sys.path.insert(0, r"E:\\Develop\\Deepseek\\tinyinfer\\src")
import torch
import tinyinfer.torch_compat  # noqa
from tinyinfer import native
N, K = 4096, 5120
w = torch.randint(0, 2**31-1, (N, K//8), dtype=torch.int32, device="cuda").view(torch.uint8)
ws = torch.full((N, K//32), 127, dtype=torch.uint8, device="cuda")
x = torch.randn(K, dtype=torch.float32, device="cuda")
asc = torch.ones(K//32, dtype=torch.float32, device="cuda")
y = torch.zeros(N, dtype=torch.bfloat16, device="cuda")
for _ in range(3):
    native.launch(native.kernel("gemv_fp4_e8m0","gemv_fp4_fast"), ((N+7)//8,), (256,), [x, asc, w, ws, y, N, K])
torch.cuda.synchronize()
print("done")
''')
ncu = r"C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.3.1\target\windows-desktop-win7-x64\ncu.exe"
cmd = [ncu, "--set", "speed-of-light", "--launch-count", "1", "--kernel-name", "gemv_fp4_fast",
       r"E:\Develop\Deepseek\tinyinfer\.venv\Scripts\python.exe", bench_py]
r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
out = r.stdout + r.stderr
# keep the interesting section
import re
keep = []
for line in out.splitlines():
    if any(k in line for k in ("Duration", "Memory Throughput", "Compute (SM)", "DRAM",
                                "L1/TEX", "L2 Cache", "Achieved Occupancy", "Registers",
                                "Issue", "Stall", "Executed Ipc", "SM Busy", "Waves")):
        keep.append(line.rstrip())
print("\n".join(keep[:45]))
