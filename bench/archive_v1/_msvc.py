"""Portability shims so vLLM's csrc builds under MSVC.

vLLM is developed on Linux; a few kernels use GCC builtins MSVC lacks. Each substitution is
exact (not an approximation) and applied to the vendored copy, with a marker comment so the
divergence stays visible.
"""
from pathlib import Path
import re

CSRC = Path(r"E:\Develop\Deepseek\repositories\vllm-1c879919786497217b3b5fb0258d1d72f2442cb0\csrc")

# a header with the MSVC equivalents
hdr = CSRC / "vl_msvc_compat.h"
hdr.write_text(r'''#pragma once
// MSVC equivalents for GCC builtins vLLM uses. Added for the Windows build; the values are
// identical to the GCC builtins, not approximations.
#ifdef _MSC_VER
#include <intrin.h>
#include <cstdint>

static inline int __builtin_clz(unsigned int x) {
    unsigned long i;
    _BitScanReverse(&i, x);
    return 31 - (int)i;
}
static inline int __builtin_ctz(unsigned int x) {
    unsigned long i;
    _BitScanForward(&i, x);
    return (int)i;
}
static inline int __builtin_popcount(unsigned int x) { return (int)__popcnt(x); }
#endif
''', encoding="utf-8")
print("wrote", hdr.name)

# inject the include into the kernels that need it
patched = 0
for f in CSRC.rglob("*.cu"):
    t = f.read_text(encoding="utf-8", errors="replace")
    if "__builtin_clz" in t or "__builtin_ctz" in t or "__builtin_popcount" in t:
        if "vl_msvc_compat.h" in t:
            continue
        # insert after the last #include at the top
        m = list(re.finditer(r'^\s*#include[^\n]*\n', t, re.M))
        if not m:
            continue
        pos = m[0].start()
        t = t[:pos] + '#include "vl_msvc_compat.h"\n' + t[pos:]
        f.write_text(t, encoding="utf-8")
        patched += 1
        print("patched", f.name)
print("total patched:", patched)
