from pathlib import Path
files = [
  Path(r"E:/Develop/Deepseek/repositories/vllm-1c879919786497217b3b5fb0258d1d72f2442cb0/csrc/quantization/marlin/kernel_selector.h"),
  Path(r"E:/Develop/Deepseek/repositories/vllm-1c879919786497217b3b5fb0258d1d72f2442cb0/csrc/moe/marlin_moe_wna16/kernel_selector.h"),
]
for p in files:
    s = p.read_text(encoding="utf-8")
    if "marlin_sel_done:;" in s:
        print(p.name, "label already present"); continue
    s = s.rstrip() + "\nmarlin_sel_done:;\n"
    p.write_text(s, encoding="utf-8")
    print(p.name, "appended label; gotos:", s.count("goto marlin_sel_done"))
# verify cutlass nan injection
p = Path(r"E:/Develop/Deepseek/repositories/vllm-1c879919786497217b3b5fb0258d1d72f2442cb0/build_cmake/_deps/cutlass-src/tools/util/include/cutlass/util/reference/host/tensor_fill.h")
s = p.read_text(encoding="utf-8", errors="replace")
print("cutlass NAN defined:", "#define NAN" in s)
print("first 3 lines:", s.splitlines()[:3])