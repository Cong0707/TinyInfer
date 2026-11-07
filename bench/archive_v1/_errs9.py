from pathlib import Path
log = Path(r"E:/Develop/Deepseek/deepseek/_bn17.log")
lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
targets = ["nvfp4_blockwise_moe_kernel.cu", "marlin.cu", "grouped_topk_kernels.cu", "marlin_moe_wna16", "ops.cu"]
for t in targets:
    idxs = [i for i,l in enumerate(lines) if l.startswith("FAILED:") and t in l]
    if not idxs:
        print("==", t, "no FAILED block (cross-target)"); continue
    idx = idxs[0]
    print("="*10, t)
    shown = 0
    for l in lines[idx+1: idx+700]:
        if "warning C4819" in l or l.startswith("tmpxft") or "C4005" in l: continue
        if "error" in l.lower():
            print("   ", l[-175:])
            shown += 1
            if shown >= 2: break