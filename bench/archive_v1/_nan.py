from pathlib import Path
p = Path(r"E:/Develop/Deepseek/repositories/vllm-1c879919786497217b3b5fb0258d1d72f2442cb0/build_cmake/_deps/cutlass-src/tools/util/include/cutlass/util/reference/host/tensor_fill.h")
s = p.read_text(encoding="utf-8", errors="replace")
if "#define NAN" not in s:
    marker = s.index("\n")
    inject = ("\n// MSVC does not expose the C99 NAN macro unconditionally; the host reference\n"
              "// helpers below need it.\n"
              "#ifndef NAN\n#define NAN (0.0f / 0.0f)\n#endif\n")
    s = s[:marker+1] + inject + s[marker+1:]
    p.write_text(s, encoding="utf-8")
    print("injected NAN into cutlass tensor_fill.h")
else:
    print("already present")