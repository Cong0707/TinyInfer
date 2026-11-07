from pathlib import Path
p = Path(r"E:/Develop/Deepseek/repositories/vllm-1c879919786497217b3b5fb0258d1d72f2442cb0/csrc/type_convert.cuh")
t = p.read_text(encoding="utf-8")
old = "  #if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800) || defined(USE_ROCM)"
new = "  // __CUDA_ARCH__ is undefined during the host pass, which made the BFloat16 specialisation"
new += "\n  // invisible there and any host-side reference to Converter::packed_hip_type a parse error."
new += "\n  // The type definition is needed in both passes; only its use is device-only."
new += "\n  #if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800 || defined(USE_ROCM)"
if old in t:
    t = t.replace(old, new, 1)
    p.write_text(t, encoding="utf-8")
    print("patched type_convert.cuh bf16 guard")
else:
    print("pattern not found")