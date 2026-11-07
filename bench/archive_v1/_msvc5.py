"""Add the remaining GCC-only spellings vLLM's NVIDIA path uses.

Each item was checked against its actual use in csrc:
- uint: used as a plain unsigned-int spelling.
- __attribute__((...)): only always_inline / __always_inline__ / __may_alias__ / target appear on
  the NVIDIA path; the vector-defining attributes (vector_size, ext_vector_type) live exclusively
  in rocm/ and cpu/ sources, which this build never compiles.
"""
from pathlib import Path

p = Path(r"E:\Develop\Deepseek\repositories\vllm-1c879919786497217b3b5fb0258d1d72f2442cb0\csrc\vl_msvc_compat.h")
t = p.read_text(encoding="utf-8")

if "typedef unsigned int uint;" not in t:
    t = t.replace(
        "// ---- GCC asm/attribute spellings -------------------------------------------------------",
        "// ---- plain C spellings ---------------------------------------------------------------\n"
        "typedef unsigned int uint;\n\n"
        "// ---- GCC asm/attribute spellings -------------------------------------------------------"
    )
    t = t.replace(
        "#define __volatile__ volatile\n#define __asm__ __asm\n",
        "#define __volatile__ volatile\n"
        "#define __asm__ __asm\n"
        "// Only inline hints / alias hints appear on the NVIDIA path; the attributes that DEFINE\n"
        "// types (vector_size, ext_vector_type) exist only under rocm/ and cpu/, never compiled here.\n"
        "#define __attribute__(x)\n"
    )
    p.write_text(t, encoding="utf-8")
    print("added uint typedef + __attribute__ elision")
else:
    print("already present")

print("--- header now ---")
for line in t.splitlines():
    if any(k in line for k in ("uint", "attribute", "volatile", "asm")):
        print(line)
