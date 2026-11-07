"""Build vLLM kernels as bare cubins by cutting off their torch host launchers.

vLLM 0.33 targets torch::stable, an ABI torch 2.9 does not have, so compiling the whole .cu
fails on the launcher even though the __global__ kernels themselves are ordinary CUDA.

So: keep everything up to the first host function that mentions torch::, write our own
C-ABI launcher for the __global__ entry points, and let native.launch drive it. The kernels
are copied verbatim -- not reimplemented -- which is the whole point of the baseline.
"""
import re, subprocess, sys
from pathlib import Path

CSRC = Path(r"E:\Develop\Deepseek\repositories\vllm-1c879919786497217b3b5fb0258d1d72f2442cb0\csrc")
TORCH_INC = Path(r"E:\Develop\Deepseek\tinyinfer\.venv\Lib\site-packages\torch\include")
SHIM = Path(r"E:\Develop\Deepseek\deepseek\_vlshim")
OUT = Path(r"E:\Develop\Deepseek\deepseek\_vlkern"); OUT.mkdir(exist_ok=True)
SRCDIR = Path(r"E:\Develop\Deepseek\deepseek\_vlsrc"); SRCDIR.mkdir(exist_ok=True)
VC = r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

def cut(src_path: Path) -> str:
    """Everything before the first host-side function that touches torch."""
    txt = src_path.read_text(encoding="utf-8", errors="replace")
    # drop vLLM's torch includes
    txt = "\n".join(l for l in txt.splitlines()
                    if not re.match(r'\s*#include\s*[<"]torch', l)
                    and "vllm/torch_utils" not in l)
    m = re.search(r'^\s*(?:void|int|__host__).*?torch::', txt, re.M)
    if m:
        txt = txt[:m.start()]
    return txt

incs = [f'-I"{SHIM}"', f'-I"{CSRC}"', f'-I"{TORCH_INC}"']
for d in sorted({p.parent for p in CSRC.rglob("*.cuh")} | {p.parent for p in CSRC.rglob("*.h")}):
    incs.append(f'-I"{d}"')
defs = ["-DUSE_CUDA=1", "-DNDEBUG", "-DFMT_HEADER_ONLY"]
base = (f'nvcc -std=c++17 --expt-relaxed-constexpr --expt-extended-lambda '
        f'-cubin -arch=sm_120 {" ".join(defs)} {" ".join(incs)}')

targets = ["layernorm_kernels.cu", "activation_kernels.cu", "cache_kernels.cu",
           "fused_qknorm_rope_kernel.cu", "sampler.cu",
           "moe/moe_align_sum_kernels.cu", "moe/topk_softmax_kernels.cu"]

for rel in targets:
    src = CSRC / rel
    if not src.exists():
        print(f"FAIL {rel}: missing", flush=True); continue
    body = cut(src)
    # a kernel body alone is not a translation unit we can name; add a dummy extern
    dummy = SRCDIR / (Path(rel).stem + ".cu")
    dummy.write_text(body + "\n__global__ void _vl_anchor() {}\n", encoding="utf-8")
    out = OUT / (Path(rel).stem + ".cubin")
    cmd = f'cmd /c "call ""{VC}"" >nul 2>&1 && {base} -o ""{out}"" ""{dummy}"""'
    r = subprocess.run(cmd, capture_output=True, text=True, shell=True, timeout=2400)
    txt = (r.stdout or "") + (r.stderr or "")
    if r.returncode == 0:
        print(f"OK   {rel}  {out.stat().st_size/1024:.0f} KB", flush=True)
    else:
        err = next((l.strip()[:120] for l in txt.splitlines() if "error" in l.lower()), "no output")
        print(f"FAIL {rel}: {err}", flush=True)
