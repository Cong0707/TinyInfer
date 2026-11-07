from pathlib import Path
p = Path(r"E:/Develop/Deepseek/repositories/vllm-1c879919786497217b3b5fb0258d1d72f2442cb0/csrc/vl_msvc_compat.h")
t = p.read_text(encoding="utf-8")
bad = ("// C99 NAN; used by cutlass' host reference helpers.\n"
       "#ifndef NAN\n#define NAN (0.0f / 0.0f)\n#endif\n"
       "#ifndef INFINITY\n#define INFINITY (1.0f / 0.0f)\n#endif\n")
if bad in t:
    t = t.replace(bad, "")
    p.write_text(t, encoding="utf-8")
    print("removed NAN/INFINITY overrides (they broke constexpr -INFINITY)")
else:
    for i, line in enumerate(t.splitlines(), 1):
        if "NAN" in line or "INFINITY" in line:
            print(i, line)