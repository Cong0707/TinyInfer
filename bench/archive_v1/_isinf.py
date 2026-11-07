from pathlib import Path
import re
base = Path(r"E:/Develop/Deepseek/repositories/vllm-1c879919786497217b3b5fb0258d1d72f2442cb0/csrc")
n = 0
for f in list(base.rglob("*.cu")) + list(base.rglob("*.cuh")):
    s = f.read_text(encoding="utf-8", errors="replace")
    if "std::isinf(" not in s and "std::isnan(" not in s:
        continue
    s2 = s.replace("std::isinf(", "::isinf(").replace("std::isnan(", "::isnan(")
    if s2 != s:
        f.write_text(s2, encoding="utf-8")
        print("fixed", f.relative_to(base))
        n += 1
print("files:", n)