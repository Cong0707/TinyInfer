src = open(r"E:\Develop\Deepseek\deepseek\_marlin_ref.cu", encoding="utf-8").read()
i = src.find("// Core loop")
if i < 0:
    i = src.find("for (int k = 0; k < k_tiles; k++)")
    if i < 0:
        i = src.find("dequant")
print(src[max(0,i-800):i+2600])
