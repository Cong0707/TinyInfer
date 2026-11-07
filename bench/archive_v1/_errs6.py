from pathlib import Path
log = Path(r"E:/Develop/Deepseek/deepseek/_bn13.log")
lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
for t in ("nvfp4_blockwise_moe_kernel.cu", "grouped_topk_kernels.cu", "topk_softplus_sqrt_kernels.cu", "selective_scan_fwd.cu"):
    idx = next((i for i,l in enumerate(lines) if l.startswith("FAILED:") and t in l), None)
    print("="*12, t)
    if idx is None: continue
    shown = 0
    for l in lines[idx+1: idx+500]:
        if "warning C4819" in l or l.startswith("tmpxft"): continue
        if "error" in l.lower():
            print("   ", l[-170:])
            shown += 1
            if shown >= 2: break