from pathlib import Path
log = Path(r"E:/Develop/Deepseek/deepseek/_bn10.log")
lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
for t in ("nvfp4_blockwise_moe_kernel.cu", "marlin.cu"):
    idx = next((i for i,l in enumerate(lines) if l.startswith("FAILED:") and t in l), None)
    print("="*15, t, "line", idx)
    if idx is None: continue
    shown = 0
    for l in lines[idx+1: idx+600]:
        if "warning C4819" in l or l.startswith("tmpxft"): continue
        low = l.lower()
        if "error" in low or "fatal" in low:
            print("   ", l[:200])
            shown += 1
            if shown >= 3: break