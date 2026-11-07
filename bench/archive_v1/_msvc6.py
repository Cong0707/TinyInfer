"""Second round of MSVC portability fixes.

1. M_SQRT2 / M_LOG2E / M_PI: glibc's <math.h> defines these by default; MSVC only does with
   _USE_MATH_DEFINES. Both the define and the fallbacks are added so either path works.
2. quant_type_max_v became a function template (variables cannot carry __host__ __device__
   under MSVC); every call site now needs the call parentheses.
"""
from pathlib import Path
import re

CSRC = Path(r"E:\Develop\Deepseek\repositories\vllm-1c879919786497217b3b5fb0258d1d72f2442cb0\csrc")

# ---- 1. math constants ------------------------------------------------------------------
p = CSRC / "vl_msvc_compat.h"
t = p.read_text(encoding="utf-8")
if "M_SQRT2" not in t:
    t = t.replace(
        "#include <cstdint>\n#include <intrin.h>\n",
        "#include <cstdint>\n#include <intrin.h>\n\n"
        "// glibc defines the M_* constants by default; MSVC needs the opt-in below.\n"
        "#ifndef _USE_MATH_DEFINES\n#define _USE_MATH_DEFINES\n#endif\n"
        "#include <cmath>\n"
        "#ifndef M_SQRT2\n#define M_SQRT2 1.41421356237309504880\n#endif\n"
        "#ifndef M_LOG2E\n#define M_LOG2E 1.44269504088896340736\n#endif\n"
        "#ifndef M_PI\n#define M_PI 3.14159265358979323846\n#endif\n"
        "#ifndef M_LN2\n#define M_LN2 0.69314718055994530942\n#endif\n"
    )
    p.write_text(t, encoding="utf-8")
    print("added math constants")
else:
    print("math constants already present")

# ---- 2. quant_type_max_v call sites -----------------------------------------------------
patched = 0
for f in CSRC.rglob("*"):
    if f.suffix not in (".cu", ".cuh", ".h", ".hpp"):
        continue
    with open(f, "r", encoding="utf-8", errors="replace") as fh:
        s = fh.read()
    if "quant_type_max_v" not in s:
        continue
    # add () where a use is not already a call
    new = re.sub(r"quant_type_max_v<([^<>]+)>(?!\()", r"quant_type_max_v<\1>()", s)
    if new != s:
        f.write_text(new, encoding="utf-8")
        patched += 1
        print("call sites fixed:", f.name)
print("files patched:", patched)
