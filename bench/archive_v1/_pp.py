from pathlib import Path
import re
CSRC = Path(r"E:\Develop\Deepseek\repositories\vllm-1c879919786497217b3b5fb0258d1d72f2442cb0\csrc")

indented = 0
files = []
for f in list(CSRC.rglob("*.cu")) + list(CSRC.rglob("*.cuh")):
    lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
    n = 0
    for l in lines:
        s = l.strip()
        if re.match(r"#\s*(ifndef|ifdef|else|endif|elif)\b", s) and (len(l) - len(l.lstrip())) > 0:
            n += 1
    if n:
        indented += n
        files.append((f.relative_to(CSRC), n))
print("indented preprocessor directives total:", indented)
for p, n in files:
    print(f"  {p}: {n}")
