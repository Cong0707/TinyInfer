src = open(r"E:\Develop\Deepseek\deepseek\_marlin_ref.cu", encoding="utf-8").read()
# find the main kernel loop and the dequant+mma core
i = src.find("__global__ void Marlin")
if i < 0:
    i = src.find("Marlingemm_kernel_switch_common")
print("=== kernel signature area ===")
print(src[i:i+1800] if i > 0 else "not found")
