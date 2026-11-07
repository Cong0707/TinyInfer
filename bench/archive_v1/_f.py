from pathlib import Path
import re
CSRC = Path(r"E:/Develop/Deepseek/repositories/vllm-1c879919786497217b3b5fb0258d1d72f2442cb0/csrc")
p = CSRC / "vl_msvc_compat.h"
t = p.read_text(encoding="utf-8")
if "define NAN" not in t:
    add = "#ifndef M_LN2\n#define M_LN2 0.69314718055994530942\n#endif\n"
    extra = "#ifndef NAN\n#define NAN (0.0f / 0.0f)\n#endif\n#ifndef INFINITY\n#define INFINITY (1.0f / 0.0f)\n#endif\n"
    t = t.replace(add, add + extra, 1)
    p.write_text(t, encoding="utf-8")
    print("added NAN/INFINITY")
def fold(path):
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    out, i, n = [], 0, 0
    while i < len(lines):
        s = lines[i].strip()
        if s == "#ifndef USE_ROCM" or re.fullmatch(r"#\s*if\s+!\s*defined\s*\(\s*USE_ROCM\s*\)", s):
            depth, j, body, alt, cur = 1, i + 1, [], [], None
            cur = body
            while j < len(lines) and depth > 0:
                st = lines[j].strip()
                if st.startswith("#if"):
                    depth += 1
                elif st.startswith("#endif"):
                    depth -= 1
                    if depth == 0: break
                elif st.startswith("#else") and depth == 1:
                    cur = alt; j += 1; continue
                cur.append(lines[j]); j += 1
            out.extend(body if body else alt)
            n += 1; i = j + 1
        else:
            out.append(lines[i]); i += 1
    if n:
        path.write_text("".join(out), encoding="utf-8")
    return n
for name in ("activation_kernels.cu", "mamba/selective_scan_fwd.cu"):
    print(name, "folded", fold(CSRC / name))