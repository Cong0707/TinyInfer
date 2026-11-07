from pathlib import Path
p = Path(r"E:\Develop\Deepseek\repositories\vllm-1c879919786497217b3b5fb0258d1d72f2442cb0\csrc\vl_msvc_compat.h")
t = p.read_text(encoding="utf-8")
t = t.replace("typedef __vl_int128 __int128_t;\ntypedef __vl_int128 __int128;\n", "typedef __vl_int128 __int128_t;\n")
p.write_text(t, encoding="utf-8")
print("dropped bare __int128 (it is a reserved MSVC token; csrc only names __int128_t)")
print([l for l in t.splitlines() if "int128" in l or "int64_t" in l])
