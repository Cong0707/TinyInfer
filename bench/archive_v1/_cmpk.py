import sys
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from pathlib import Path

b = Path(r"E:\Develop\Deepseek\tinyinfer\src\tinyinfer\native\gemv_bf16.cu").read_text(encoding="utf-8")
i = b.find("__global__")
print("=== gemv_bf16 (309 GB/s, 92% peak) ===")
print(b[i:i+1800])
