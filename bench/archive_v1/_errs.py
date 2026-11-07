from pathlib import Path
import re
log = Path(r"E:\Develop\Deepseek\deepseek\_bn8.log")
text = log.read_text(encoding="utf-8", errors="replace")
# error lines carry the full source path; group by the file they name
targets = ["activation_kernels.cu", "merge_attn_states.cu", "marlin.cu",
           "nvfp4_blockwise_moe_kernel.cu", "selective_scan_fwd.cu",
           "layernorm_quant_kernels.cu", "fused_layernorm_dynamic_per_token_quant.cu",
           "fused_silu_mul_block_quant.cu", "gemm_kernels.cu", "common.cu",
           "fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu"]
errs = [l for l in text.splitlines() if "error:" in l]
for t in targets:
    hits = [l for l in errs if t in l]
    print("===", t, ":", len(hits), "errors")
    for h in hits[:2]:
        msg = h.split("error:")[-1].strip()
        src = h.split("csrc")[-1].split("(")[0] if "csrc" in h else ""
        print("   ", src, "->", msg[:110])
