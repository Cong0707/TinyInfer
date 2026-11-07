from pathlib import Path
p = Path(r"E:/Develop/Deepseek/repositories/vllm-1c879919786497217b3b5fb0258d1d72f2442cb0/build_cmake/_deps/cutlass-src/tools/util/include/cutlass/util/reference/host/tensor_fill.h")
s = p.read_text(encoding="utf-8")
old = ("// MSVC does not expose the C99 NAN macro unconditionally; cutlass' host\n"
       "// reference helpers use it below.\n"
       "#ifndef NAN\n#define NAN (0.0f / 0.0f)\n#endif\n")
new = ("// MSVC does not expose the C99 NAN macro unconditionally, and the usual\n"
       "// 0.0f/0.0f spelling is a compile error (C2124) under cl. quiet_NaN() is a\n"
       "// constant expression with the same value.\n"
       "#include <limits>\n"
       "#undef NAN\n"
       "#define NAN (std::numeric_limits<float>::quiet_NaN())\n")
if old in s:
    s = s.replace(old, new)
    p.write_text(s, encoding="utf-8")
    print("cutlass NAN -> numeric_limits")
else:
    print("old block not found; head is:"); print("\n".join(s.splitlines()[:8]))

# same treatment for the vllm fp4 kernel
p2 = Path(r"E:/Develop/Deepseek/repositories/vllm-1c879919786497217b3b5fb0258d1d72f2442cb0/csrc/quantization/fp4/nvfp4_blockwise_moe_kernel.cu")
s2 = p2.read_text(encoding="utf-8")
old2 = ("// cutlass' host reference helpers use the C99 NAN macro; MSVC does not\n"
        "// expose it without <math.h>.\n"
        "#ifndef NAN\n#define NAN (0.0f / 0.0f)\n#endif\n")
new2 = ("// cutlass' host reference helpers need NAN; see tensor_fill.h for why this\n"
        "// uses quiet_NaN rather than the 0.0f/0.0f spelling.\n"
        "#include <limits>\n"
        "#ifndef NAN\n#define NAN (std::numeric_limits<float>::quiet_NaN())\n#endif\n")
if old2 in s2:
    s2 = s2.replace(old2, new2)
    p2.write_text(s2, encoding="utf-8")
    print("kernel NAN -> numeric_limits")
else:
    print("kernel block not found")