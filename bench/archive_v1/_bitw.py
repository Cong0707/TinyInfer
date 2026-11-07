"""Replace GCC's __builtin_clz in the constexpr helper with C++20 std::bit_width.

next_pow_2 is constexpr, so the MSVC _BitScanReverse shim (not constexpr) cannot stand in.
std::bit_width is the exact C++20 spelling of "CHAR_BIT * sizeof(x) - __builtin_clz(x)" and is
constexpr in every implementation.
"""
from pathlib import Path

p = Path(r"E:\Develop\Deepseek\repositories\vllm-1c879919786497217b3b5fb0258d1d72f2442cb0\csrc\core\utils.hpp")
t = p.read_text(encoding="utf-8")

if "std::bit_width" in t:
    print("already patched")
else:
    t = t.replace("#include <climits>", "#include <bit>\n#include <climits>", 1)
    old = "  return 1 << (CHAR_BIT * sizeof(num) - __builtin_clz(num - 1));"
    new = "  // std::bit_width is the C++20 constexpr equivalent of\n" \
          "  // (CHAR_BIT * sizeof(T) - __builtin_clz(v)); MSVC has no __builtin_clz.\n" \
          "  return 1 << std::bit_width(num - 1);"
    assert old in t, "pattern not found"
    t = t.replace(old, new, 1)
    p.write_text(t, encoding="utf-8")
    print("patched utils.hpp")
