from pathlib import Path
import re
log = Path(r"E:/Develop/Deepseek/deepseek/_bn10.log")
lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
errs = [l for l in lines if "error:" in l or "fatal error" in l]
targets = ["fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu", "selective_scan_fwd.cu",
           "gemm_kernels.cu", "nvfp4_blockwise_moe_kernel.cu", "marlin.cu"]
for t in targets:
    hits = [l for l in errs if t in l]
    print("===", t, len(hits))
    seen = set()
    for h in hits:
        msg = h.split("error:")[-1].strip()[:95]
        if msg in seen: continue
        seen.add(msg)
        m = re.search(r"csrc[\\/]([^(]+)\((\d+)\)", h)
        loc = (m.group(1) + ":" + m.group(2)) if m else ""
        print("   ", loc[:48], "|", msg)
        if len(seen) >= 3: break