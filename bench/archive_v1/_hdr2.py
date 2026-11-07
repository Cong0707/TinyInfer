from pathlib import Path
p = Path(r"E:/Develop/Deepseek/repositories/vllm-1c879919786497217b3b5fb0258d1d72f2442cb0/csrc/vl_msvc_compat.h")
t = p.read_text(encoding="utf-8")
bad = "#if _MSC_VER\n#include <cmath>\n#define _USE_MATH_DEFINES 1\n#endif\n#include <cmath>\n"
good = "// _USE_MATH_DEFINES must precede <cmath> for MSVC to expose the M_* constants.\n"
good += "#ifndef _USE_MATH_DEFINES\n#define _USE_MATH_DEFINES\n#endif\n#include <cmath>\n"
if bad in t:
    t = t.replace(bad, good, 1)
    p.write_text(t, encoding="utf-8")
    print("fixed _USE_MATH_DEFINES order")
else:
    print("pattern not found")