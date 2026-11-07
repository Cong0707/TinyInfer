from pathlib import Path
import re
files = [
  Path(r"E:/Develop/Deepseek/repositories/vllm-1c879919786497217b3b5fb0258d1d72f2442cb0/csrc/quantization/marlin/kernel_selector.h"),
  Path(r"E:/Develop/Deepseek/repositories/vllm-1c879919786497217b3b5fb0258d1d72f2442cb0/csrc/moe/marlin_moe_wna16/kernel_selector.h"),
]
pat = re.compile(r"^(?:else )?if (\(.*\))\n  (kernel = .*;)$", re.M)
for p in files:
    s = p.read_text(encoding="utf-8")
    n_before = s.count("else if (")
    s2, n = pat.subn(r"if (\1) { \2 goto marlin_sel_done; }", s)
    if "marlin_sel_done" not in s2.splitlines()[-1]:
        s2 = s2.rstrip() + "\nmarlin_sel_done:;\n"
    p.write_text(s2, encoding="utf-8")
    print(p.name, "else-if:", n_before, "-> converted:", n, "remaining else:", s2.count("else if ("))