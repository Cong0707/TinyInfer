from pathlib import Path
import re
p = Path(r"E:/Develop/Deepseek/repositories/vllm-1c879919786497217b3b5fb0258d1d72f2442cb0/csrc/quantization/awq/gemm_kernels.cu")
t = p.read_text(encoding="utf-8")
n = t.count("asm volatile(")
t2 = t.replace("asm volatile(", "__asm__ __volatile__(")
if t2 != t:
    p.write_text(t2, encoding="utf-8")
print("replaced", n, "asm volatile( -> __asm__ __volatile__(")
# also check for bare asm(
print("remaining bare asm(:", len(re.findall(r"(?<!_)\basm\s*\(", t2)))