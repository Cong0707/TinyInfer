"""Windows/MSVC portability layer for vLLM's csrc.

vLLM targets Linux/GCC. The kernels do not need GCC: every construct below is a spelling or
type that MSVC lacks, and each replacement has identical semantics. Nothing here changes a
kernel's arithmetic.
"""
from pathlib import Path

CSRC = Path(r"E:\Develop\Deepseek\repositories\vllm-1c879919786497217b3b5fb0258d1d72f2442cb0\csrc")

header = r'''#pragma once
// Portability layer so vLLM's csrc builds under MSVC (Windows). vLLM targets Linux/GCC; each
// item below is a GCC-only spelling or builtin that MSVC lacks. None of them changes what a
// kernel computes. Force-included into every CUDA translation unit.
#ifdef _MSC_VER

#include <cstdint>
#include <intrin.h>

// ---- GCC integer builtins -------------------------------------------------------------
static __inline int __builtin_clz(unsigned int x) {
  unsigned long i; _BitScanReverse(&i, x); return 31 - (int)i;
}
static __inline int __builtin_ctz(unsigned int x) {
  unsigned long i; _BitScanForward(&i, x); return (int)i;
}
static __inline int __builtin_popcount(unsigned int x) { return (int)__popcnt(x); }
static __inline int __builtin_clzll(unsigned long long x) {
  unsigned long i; _BitScanReverse64(&i, x); return 63 - (int)i;
}

// ---- 128/64-bit 'extended' integer names -----------------------------------------------
// vLLM uses these only as 16-/8-byte STORAGE units for shared-memory addressing and vector
// loads; there is no 128-bit arithmetic on them anywhere in csrc. A pair of 64-bit words is
// therefore an exact stand-in, including sizeof and alignment.
struct __vl_int128 { uint64_t lo, hi; };
typedef __vl_int128 __int128_t;
typedef __vl_int128 __int128;
typedef long long __int64_t;
typedef unsigned long long __uint64_t;

// ---- GCC asm/attribute spellings -------------------------------------------------------
#define __volatile__ volatile
#define __asm__ __asm
#define __align__(n) __declspec(align(n))

#endif  // _MSC_VER
'''
(CSRC / "vl_msvc_compat.h").write_text(header, encoding="utf-8")
print("wrote vl_msvc_compat.h")

# quant_type_max_v: nvcc rejects __host__ __device__ on a variable template under MSVC.
# A function template is the idiomatic equivalent and is accepted everywhere.
p = CSRC / "quantization" / "utils.cuh"
t = p.read_text(encoding="utf-8")
if "quant_type_max_v()" not in t:
    old = """template <typename T>
MAYBE_HOST_DEVICE static constexpr T quant_type_max_v =
    quant_type_max<T>::val();"""
    new = """template <typename T>
MAYBE_HOST_DEVICE static constexpr T quant_type_max_v() {
  // Was a __host__ __device__ variable template; MSVC's nvcc rejects the attribute on
  // variables, and a function template is the equivalent spelling.
  return quant_type_max<T>::val();
}"""
    assert old in t, "quant_type_max_v pattern not found"
    t = t.replace(old, new, 1)
    t = t.replace("quant_type_max_v<T> * 512.0f", "quant_type_max_v<T>() * 512.0f")
    p.write_text(t, encoding="utf-8")
    print("patched quant_type_max_v -> function")
else:
    print("quant_type_max_v already patched")
