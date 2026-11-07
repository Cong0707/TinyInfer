from pathlib import Path
CSRC = Path(r"E:\Develop\Deepseek\repositories\vllm-1c879919786497217b3b5fb0258d1d72f2442cb0\csrc")
p = CSRC / "vl_msvc_compat.h"
t = p.read_text(encoding="utf-8")
t = t.replace("""#define __volatile__ volatile
#define __asm__ __asm
#define __align__(n) __declspec(align(n))
""", """#define __volatile__ volatile
#define __asm__ __asm
""")
p.write_text(t, encoding="utf-8")
print("removed __align__ override (it is a CUDA builtin; the error was the type, not the attribute)")
