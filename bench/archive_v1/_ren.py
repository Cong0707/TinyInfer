from pathlib import Path
import re
base = Path(r"E:\Develop\Deepseek\repositories\vllm-1c879919786497217b3b5fb0258d1d72f2442cb0\csrc\quantization\fp4")
for name in ("nvfp4_experts_quant.cu", "mxfp4_experts_quant.cu"):
    f = base / name
    if not f.exists():
        print("skip", name); continue
    t = f.read_text(encoding="utf-8")
    if "ST_FLOAT" in t:
        print("already", name); continue
    for old, new in (("FLOAT", "ST_FLOAT"), ("INT", "ST_INT"), ("UINT8", "ST_UINT8")):
        t = re.sub(r"constexpr auto " + old + r"\b", "constexpr auto " + new, t)
        t = re.sub(r"\b" + old + r"\b", new, t)
    f.write_text(t, encoding="utf-8")
    print("renamed", name)