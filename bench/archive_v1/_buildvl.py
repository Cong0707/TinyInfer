import os, subprocess
from pathlib import Path

CSRC = Path(r"E:\Develop\Deepseek\repositories\vllm-1c879919786497217b3b5fb0258d1d72f2442cb0\csrc")
TORCH_INC = Path(r"E:\Develop\Deepseek\tinyinfer\.venv\Lib\site-packages\torch\include")
OUT = Path(r"E:\Develop\Deepseek\deepseek\_vlkern"); OUT.mkdir(exist_ok=True)
VC = r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

incs = [f'-I"{CSRC}"', f'-I"{TORCH_INC}"',
        f'-I"{TORCH_INC / "torch" / "csrc" / "api" / "include"}"']
for d in sorted({p.parent for p in CSRC.rglob("*.cuh")} | {p.parent for p in CSRC.rglob("*.h")}):
    incs.append(f'-I"{d}"')

defs = ["-DTORCH_EXTENSION_NAME=vllm", "-DVLLM_USE_V1=1", "-DUSE_CUDA=1",
        "-DENABLE_SCALED_MM_SM120=1", "-DNDEBUG", "-DFMT_HEADER_ONLY"]
base = (f'nvcc -std=c++17 --expt-relaxed-constexpr --expt-extended-lambda '
        f'-cubin -arch=sm_120 {" ".join(defs)} {" ".join(incs)}')

targets = [
 "layernorm_kernels.cu", "activation_kernels.cu", "cache_kernels.cu",
 "cache_kernels_fused.cu", "fused_qknorm_rope_kernel.cu",
 "fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu", "sampler.cu",
 "moe/moe_align_sum_kernels.cu", "moe/moe_permute_unpermute_op.cu",
 "moe/grouped_topk_kernels.cu", "moe/topk_softmax_kernels.cu",
 "moe/moe_wna16.cu", "moe/marlin_moe_wna16/ops.cu",
]

ok, bad = [], []
for rel in targets:
    src = CSRC / rel
    if not src.exists():
        bad.append((rel, "missing")); continue
    out = OUT / (Path(rel).stem + ".cubin")
    cmd = f'cmd /c "call ""{VC}"" >nul 2>&1 && {base} -o ""{out}"" ""{src}"""'
    r = subprocess.run(cmd, capture_output=True, text=True, shell=True, timeout=2400)
    txt = r.stdout + r.stderr
    if r.returncode == 0:
        ok.append(rel); print(f"OK   {rel}  {out.stat().st_size/1024:.0f} KB", flush=True)
    else:
        err = next((l.strip()[:130] for l in txt.splitlines() if "error" in l.lower()),
                   txt.strip().splitlines()[-1][:130] if txt.strip() else "no output")
        bad.append((rel, err)); print(f"FAIL {rel}: {err}", flush=True)

print(f"\nbuilt {len(ok)}/{len(targets)}")
