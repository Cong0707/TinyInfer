from pathlib import Path
# 1. nvfp4 kernel: define NAN right after the license block, before any include.
p = Path(r"E:/Develop/Deepseek/repositories/vllm-1c879919786497217b3b5fb0258d1d72f2442cb0/csrc/quantization/fp4/nvfp4_blockwise_moe_kernel.cu")
s = p.read_text(encoding="utf-8")
if "#define NAN" not in s:
    marker = "#include <torch/csrc/stable/library.h>"
    inject = ("// cutlass' host reference helpers use the C99 NAN macro; MSVC does not\n"
              "// expose it without <math.h>.\n"
              "#ifndef NAN\n#define NAN (0.0f / 0.0f)\n#endif\n\n")
    s = s.replace(marker, inject + marker, 1)
    p.write_text(s, encoding="utf-8")
    print("nvfp4: injected NAN")
else:
    print("nvfp4: already has NAN")

# 2. grouped_topk: -INFINITY is not a constant expression nvcc accepts.
p2 = Path(r"E:/Develop/Deepseek/repositories/vllm-1c879919786497217b3b5fb0258d1d72f2442cb0/csrc/moe/grouped_topk_kernels.cu")
s2 = p2.read_text(encoding="utf-8")
old = "static constexpr float InvalidScore = -INFINITY;"
new = "static constexpr float InvalidScore = -std::numeric_limits<float>::infinity();"
if old in s2:
    s2 = s2.replace(old, new)
    if "#include <limits>" not in s2:
        first_inc = s2.index("#include")
        s2 = s2[:first_inc] + "#include <limits>\n" + s2[first_inc:]
    p2.write_text(s2, encoding="utf-8")
    print("grouped_topk: InvalidScore fixed")
else:
    print("grouped_topk: pattern not found")