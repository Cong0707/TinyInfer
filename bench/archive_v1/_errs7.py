from pathlib import Path
log = Path(r"E:/Develop/Deepseek/deepseek/_bn14.log")
lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
idx = next((i for i,l in enumerate(lines) if l.startswith("FAILED:") and "selective_scan" in l), None)
print("idx", idx)
shown = 0
for l in lines[idx+1: idx+900]:
    if "warning C4819" in l or l.startswith("tmpxft"): continue
    if "error" in l.lower():
        print("   ", l[-190:])
        shown += 1
        if shown >= 5: break