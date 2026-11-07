from pathlib import Path
p = Path(r"E:/Develop/Deepseek/repositories/vllm-1c879919786497217b3b5fb0258d1d72f2442cb0/csrc/quantization/awq/gemm_kernels.cu")
s = p.read_text(encoding="utf-8")
n = s.count("__asm__ __volatile__")
s = s.replace("__asm__ __volatile__", "asm volatile")
p.write_text(s, encoding="utf-8")
print("reverted", n, "to asm volatile; remaining __asm__:", s.count("__asm__"))