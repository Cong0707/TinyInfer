import subprocess, os, sys
os.chdir(r"E:\Develop\Deepseek\deepseek")
vc = r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
cu = open("_deqdump.cu").read()
cu = cu.replace("void deq_dump(", "void deq_dump8(")
cu = cu.replace("""    out[i * 4 + 0] = o[0].x;""", """    __half2 p[2];
    deq4((int)((unsigned int)w[i] >> 16), p);
    out[i * 8 + 0] = o[0].x;""")
cu = cu.replace("""    out[i * 4 + 3] = o[1].y;""", """    out[i * 8 + 3] = o[1].y;
    out[i * 8 + 4] = p[0].x;
    out[i * 8 + 5] = p[0].y;
    out[i * 8 + 6] = p[1].x;
    out[i * 8 + 7] = p[1].y;""")
cu = cu.replace("out[i * 4 + 1]", "out[i * 8 + 1]").replace("out[i * 4 + 2]", "out[i * 8 + 2]")
open("_deqdump8.cu", "w").write(cu)
r = subprocess.run(
    'cmd /c "call "' + vc + '" >nul 2>&1 && nvcc -cubin -arch=sm_120 -o _deqdump8.cubin _deqdump8.cu"',
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
    tests = [("nibbles 0..7", 0), ("nibbles 1..8", 0x87654321)]
    w0 = 0
    for i in range(8): w0 |= (i & 0xF) << (4*i)
    tests[0] = ("nibbles 0,1,2,3,4,5,6,7", w0)
    for name, word in tests:
        w = torch.tensor([word], dtype=torch.int32, device="cuda")
        out = torch.zeros(8, dtype=torch.float16, device="cuda")
        N.launch(N.kernel("_deqdump8","deq_dump8"), (1,), (1,), [w, out, 1])
        torch.cuda.synchronize()
        print(f"{name}: {out.tolist()}")
finally:
    N._DIR = orig