from pathlib import Path
p = Path(r"E:/Develop/Deepseek/repositories/vllm-1c879919786497217b3b5fb0258d1d72f2442cb0/build_cmake/_deps/cutlass-src/tools/util/include/cutlass/util/reference/host/tensor_fill.h")
s = p.read_text(encoding="utf-8")
bad = ("// MSVC does not expose the C99 NAN macro unconditionally; the host reference\n"
       "// helpers below need it.\n"
       "#ifndef NAN\n#define NAN (0.0f / 0.0f)\n#endif\n")
if bad in s:
    s = s.replace(bad, "")
    print("removed misplaced injection")
if not s.startswith("#ifndef NAN"):
    s = ("// MSVC does not expose the C99 NAN macro unconditionally; cutlass' host\n"
         "// reference helpers use it below.\n"
         "#ifndef NAN\n#define NAN (0.0f / 0.0f)\n#endif\n") + s
    p.write_text(s, encoding="utf-8")
    print("injected at file start")
print("first line:", s.splitlines()[0])
print("has define:", "#define NAN" in s)