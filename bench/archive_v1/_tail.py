from pathlib import Path
src = Path(r"E:\Develop\Deepseek\deepseek\_marlin_src\marlin_cuda_kernel.cu").read_text(encoding="utf-8")
print("=== tail 2500 ===")
print(src[-2500:])
