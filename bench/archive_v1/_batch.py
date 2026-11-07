import re, subprocess
from pathlib import Path

CSRC = Path(r"E:\Develop\Deepseek\repositories\vllm-1c879919786497217b3b5fb0258d1d72f2442cb0\csrc")
SHIM = Path(r"E:\Develop\Deepseek\deepseek\_vlshim")
OUT = Path(r"E:\Develop\Deepseek\deepseek\_vlkern"); OUT.mkdir(exist_ok=True)
SRCDIR = Path(r"E:\Develop\Deepseek\deepseek\_vlsrc"); SRCDIR.mkdir(exist_ok=True)
VC = r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

def cut(src_path: Path) -> str:
    txt = src_path.read_text(encoding="utf-8", errors="replace")
    txt = "\n".join(l for l in txt.splitlines()
                    if not re.match(r'\s*#include\s*[<"]torch', l)
                    and "torch_utils" not in l)
    m = re.search(r'^\s*(?:void|int|__host__).*?torch::', txt, re.M)
    if m:
        txt = txt[:m.start()]
    return txt

targets = ["activation_kernels.cu", "cache_kernels.cu", "cache_kernels_fused.cu",
           "fused_qknorm_rope_kernel.cu", "sampler.cu",
           "moe/moe_align_sum_kernels.cu", "moe/topk_softmax_kernels.cu",
           "moe/grouped_topk_kernels.cu", "fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu"]

inc = f'-I"{SHIM}" -I"{CSRC}" -I"{CSRC / "core"}"'
base = f'nvcc -std=c++17 --expt-relaxed-constexpr -cubin -arch=sm_120 -DUSE_CUDA=1 -DNDEBUG {inc}'

for rel in targets:
    src = CSRC / rel
    if not src.exists():
        print(f"MISS {rel}", flush=True); continue
    dummy = SRCDIR / (Path(rel).stem + ".cu")
    dummy.write_text(cut(src) + "\n__global__ void _vl_anchor() {}\n", encoding="utf-8")
    out = OUT / (Path(rel).stem + ".cubin")
    cl = f'"{VC}" >nul 2>&1 && {base} -o "{out}" "{dummy}"'
    r = subprocess.run(f'cmd /c {cl}', capture_output=True, text=True, shell=True, timeout=2400)
    txt = (r.stdout or "") + (r.stderr or "")
    if r.returncode == 0:
        print(f"OK   {rel}  {out.stat().st_size/1024:.0f} KB", flush=True)
    else:
        err = next((l.strip()[:120] for l in txt.splitlines() if "error" in l.lower()), "no output")
        print(f"FAIL {rel}: {err}", flush=True)
