from pathlib import Path
p = Path(r"E:/Develop/Deepseek/repositories/vllm-1c879919786497217b3b5fb0258d1d72f2442cb0/csrc/vl_msvc_compat.h")
t = p.read_text(encoding="utf-8")
mathblock = t[t.index("#ifndef _USE_MATH_DEFINES"):t.index("// ---- GCC integer builtins")]
mathblock_clean = mathblock.replace("#ifndef _USE_MATH_DEFINES\n#define _USE_MATH_DEFINES\n#endif\n", "#if _MSC_VER\n#include <cmath>\n#define _USE_MATH_DEFINES 1\n#endif\n")
t = t.replace(mathblock, "")
# put the constants before the _MSC_VER guard so every pass sees them
guard = "#ifdef _MSC_VER\n\n#include <cstdint>\n#include <intrin.h>\n"
newguard = mathblock_clean + "\n#ifdef _MSC_VER\n\n#include <cstdint>\n#include <intrin.h>\n"
t = t.replace(guard, newguard, 1)
p.write_text(t, encoding="utf-8")
print("moved math constants out of the _MSC_VER guard")
print("---head---")
print("\n".join(t.splitlines()[:30]))