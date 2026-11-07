"""Transform a vLLM .cu into one that compiles against a POD Tensor stub.

The kernels must stay verbatim; only the host-side plumbing changes. Rules are textual and
narrow, so anything unexpected shows up as a compile error rather than silent breakage.
"""
import re
from pathlib import Path

CSRC = Path(r"E:\Develop\Deepseek\repositories\vllm-1c879919786497217b3b5fb0258d1d72f2442cb0\csrc")
SRCDIR = Path(r"E:\Develop\Deepseek\deepseek\_vlsrc"); SRCDIR.mkdir(exist_ok=True)

def transform(src_path: Path) -> str:
    t = src_path.read_text(encoding="utf-8", errors="replace")
    # 1. swap the torch includes for our stub + the shim headers
    t = re.sub(r'\s*#include\s*[<"]torch/csrc/[^>"]*[>"]', '', t)
    t = re.sub(r'\s*#include\s*[<"]torch/headeronly/[^>"]*[>"]', '', t)
    t = re.sub(r'\s*#include\s*[<"]torch_utils\.h[>"]', '', t)
    # 2. torch::stable::Tensor & helpers -> our POD
    t = t.replace("torch::stable::Tensor", "torch::stable::Tensor")
    t = t.replace("torch::stable::contiguous", "torch::stable::contiguous")
    t = t.replace("torch::stable::accelerator::DeviceGuard", "torch::stable::accelerator::DeviceGuard")
    t = t.replace("get_current_cuda_stream()", "torch::stable::get_current_cuda_stream()")
    # 3. prepend the stub
    head = '#include "tensor_stub.h"\n#include "dispatch_utils.h"\n'
    return head + t

targets = ["layernorm_kernels.cu", "activation_kernels.cu", "cache_kernels.cu",
           "moe/moe_align_sum_kernels.cu"]
for rel in targets:
    src = CSRC / rel
    out = SRCDIR / (Path(rel).stem + "_full.cu")
    out.write_text(transform(src), encoding="utf-8")
    print("wrote", out.name, out.stat().st_size, "bytes")
