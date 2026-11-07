from pathlib import Path
import re
log = Path(r"E:\Develop\Deepseek\deepseek\_bn9.log")
lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
# find the FAILED block for these two and print the first few non-noise lines after it
for t in ("marlin.cu", "nvfp4_blockwise_moe_kernel.cu"):
    idx = None
    for i, l in enumerate(lines):
        if l.startswith("FAILED:") and t in l:
            idx = i
            break
    print("=" * 20, t, "at line", idx)
    if idx is None:
        continue
    shown = 0
    for l in lines[idx + 1: idx + 400]:
        if "warning C4819" in l or l.startswith("tmpxft") or l.strip().endswith(".cu"):
            continue
        if "error" in l.lower() or "fatal" in l.lower() or "FAILED" in l:
            print("   ", l[:220])
            shown += 1
            if shown >= 4:
                break
