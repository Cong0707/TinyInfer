"""Extract and instantiate Marlin for our shape, then benchmark it next to my kernel."""
import re
from pathlib import Path

src = Path(r"E:\Develop\Deepseek\deepseek\_marlin_src\marlin_cuda_kernel.cu").read_text(encoding="utf-8")
# find the launch macros
i = src.find("#define MARLIN_LAUNCH")
if i < 0:
    i = src.find("cudaError_t marlin_cuda")
print(src[i:i+2200] if i > 0 else "launch area not found")
print()
print("=== template params ===")
j = src.find("template <")
print(src[j:j+600])
