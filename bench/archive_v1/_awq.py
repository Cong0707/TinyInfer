from pathlib import Path
import re
p = Path(r"E:/Develop/Deepseek/repositories/vllm-1c879919786497217b3b5fb0258d1d72f2442cb0/csrc/quantization/awq/gemm_kernels.cu")
s = p.read_text(encoding="utf-8")

sub_pat = re.compile(
    r'__asm__\s+__volatile__\("sub\.f16x2 %0, %1, %2;\\n"'
    r'\s*:\s*"=r"\(([^()]+)\)'
    r'\s*:\s*"r"\(([^()]+)\),\s*"r"\(([^()]+)\)\s*\);',
    re.S)
fma_pat = re.compile(
    r'__asm__\s+__volatile__\("fma\.rn\.f16x2 %0, %1, %2, %3;\\n"'
    r'\s*:\s*"=r"\(([^()]+)\)'
    r'\s*:\s*"r"\(([^()]+)\),\s*"r"\(([^()]+)\),\s*"r"\(([^()]+)\)\s*\);',
    re.S)

s, n_sub = sub_pat.subn(r"\1 = __half2_as_uint(__hsub2(__uint_as_half2(\2), __uint_as_half2(\3)));", s)
s, n_fma = fma_pat.subn(r"\1 = __half2_as_uint(__hfma2(__uint_as_half2(\2), __uint_as_half2(\3), __uint_as_half2(\4)));", s)
print("sub replaced", n_sub, "fma replaced", n_fma)
print("remaining __asm__:", s.count("__asm__"))
if s.count("__asm__") == 0:
    p.write_text(s, encoding="utf-8")
    print("written")
else:
    import re as _re
    for m in _re.finditer(r"__asm__[^;]{0,200};", s):
        print("LEFT:", m.group(0)[:160].replace(chr(10), " "))