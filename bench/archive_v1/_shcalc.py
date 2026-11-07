"""Compile vLLM's Marlin as a bare cubin: the kernels need no torch, only the launcher does.

marlin.cu = torch bindings (skip). marlin_template.h = the kernel (use).
Strategy: write a small launcher TU that instantiates the templates for the M=1 shape and
exposes __global__ entry points with plain C-ABI signatures our native.launch can drive.
Then sanity-check every buffer size against the kernel's own index math before launching.
"""
from pathlib import Path
d = Path(r"E:\Develop\Deepseek\deepseek\_marlin_src")

# compute the exact shared-memory requirement from the kernel's constants
src = (d / "marlin_template.h").read_text(encoding="utf-8")
import re
def grab(name):
    m = re.search(rf"constexpr int {name}\s*=\s*([^;]+);", src)
    return m.group(1).strip() if m else None
for n in ("a_sh_stride", "a_sh_stage", "b_sh_stride", "b_sh_stage", "a_sh_wr_iters", "b_sh_wr_iters"):
    print(n, "=", grab(n))
