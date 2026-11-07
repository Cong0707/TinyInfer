from pathlib import Path
import re
log = Path(r"E:\Develop\Deepseek\deepseek\_bn9.log")
lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
errs = [l for l in lines if "error:" in l]
targets = ["activation_kernels.cu", "merge_attn_states.cu", "marlin.cu",
           "nvfp4_blockwise_moe_kernel.cu", "selective_scan_fwd.cu",
           "gemm_kernels.cu", "fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu"]
for t in targets:
    hits = [l for l in errs if t in l]
    print(f"=== {t}: {len(hits)}")
    seen = set()
    for h in hits:
        msg = h.split("error:")[-1].strip()[:100]
        if msg in seen:
            continue
        seen.add(msg)
        loc = ""
        m = re.search(r"csrc[\\/]([^(]+)\((\d+)\)", h)
        if m:
            loc = f"{m.group(1)}:{m.group(2)}"
        print(f"    {loc:45s} {msg}")
        if len(seen) >= 3:
            break
