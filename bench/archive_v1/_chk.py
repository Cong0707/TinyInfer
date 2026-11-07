from pathlib import Path
import re
f = Path(r"E:\Develop\Deepseek\repositories\vllm-1c879919786497217b3b5fb0258d1d72f2442cb0\csrc\quantization\fp4\nvfp4_experts_quant.cu")
t = f.read_text(encoding="utf-8")
for n in ("FLOAT", "INT", "UINT8"):
    print(n, len(re.findall(r"\b" + n + r"\b", t)), "uses")
# which other files declare these
csrc = Path(r"E:\Develop\Deepseek\repositories\vllm-1c879919786497217b3b5fb0258d1d72f2442cb0\csrc")
for p in csrc.rglob("*.cu"):
    s = p.read_text(encoding="utf-8", errors="replace")
    if re.search(r"constexpr auto (FLOAT|INT|UINT8)\b", s):
        print("declares in:", p.relative_to(csrc))
