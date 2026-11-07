from pathlib import Path
import re
base = Path(r"E:/Develop/Deepseek/repositories/vllm-1c879919786497217b3b5fb0258d1d72f2442cb0/csrc")
pat = re.compile(r"__attribute_?\(\s*\(\s*aligned\s*\(\s*(\d+)\s*\)\s*\)\s*\)")
n = 0
for f in list(base.rglob("*.cu")) + list(base.rglob("*.cuh")) + list(base.rglob("*.h")):
    s = f.read_text(encoding="utf-8", errors="replace")
    if "__attribute" not in s:
        continue
    s2 = pat.sub(r"__align__(\1)", s)
    if s2 != s:
        f.write_text(s2, encoding="utf-8")
        print("aligned->__align__:", f.relative_to(base))
        n += 1
print("files:", n)