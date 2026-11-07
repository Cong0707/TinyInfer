from pathlib import Path
p = Path(r"E:/Develop/Deepseek/repositories/vllm-1c879919786497217b3b5fb0258d1d72f2442cb0/csrc/vl_msvc_compat.h")
t = p.read_text(encoding="utf-8")
old = "#define __volatile__ volatile\n#define __asm__ __asm\n"
if old in t:
    t = t.replace(old, "// __asm__/__volatile__ are intentionally NOT redefined: nvcc's device front end\n"
                       "// understands the GCC spelling, and mapping them to MSVC's __asm turns valid\n"
                       "// device inline PTX into a Microsoft-style asm declaration error.\n")
    p.write_text(t, encoding="utf-8")
    print("removed __asm__/__volatile__ overrides")
else:
    print("pattern not found; current header:")
    for line in t.splitlines():
        if "asm" in line or "volatile" in line:
            print("   ", line)