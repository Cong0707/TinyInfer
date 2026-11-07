"""Point the engine at gemv_fp4_fast (68 GB/s) instead of the failed variants."""
from pathlib import Path
p = Path(r"E:\Develop\Deepseek\tinyinfer\src\tinyinfer\gemv_fp4.py")
src = p.read_text(encoding="utf-8")
# the wiring already calls gemv_fp4_fast; verify it is present
print("calls gemv_fp4_fast:", "gemv_fp4_fast" in src)
print("has native gate:", "_native_available" in src)
