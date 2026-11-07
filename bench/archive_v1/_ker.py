from pathlib import Path
p = Path(r"E:\Develop\Deepseek\tinyinfer\src\tinyinfer\native\gemv_fp4_e8m0.cu")
src = p.read_text(encoding="utf-8")
i = src.find("__global__")
print(src[i:i+2600])
