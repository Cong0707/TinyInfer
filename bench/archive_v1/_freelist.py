"""Which vLLM kernels compile without torch? Check their includes.

A kernel that only includes cuda_runtime + std can be built into a cubin and driven by our
native.launch. Those that include torch/* need the full extension build and are skipped for
the baseline (or wrapped later).
"""
from pathlib import Path
root = Path(r"E:\Develop\Deepseek\repositories\vllm-csrc\csrc")
rows = []
for f in sorted(root.rglob("*.cu")):
    rel = str(f.relative_to(root))
    if "rocm" in rel or "hip" in rel.lower():
        continue
    txt = f.read_text(encoding="utf-8", errors="replace")
    inc = [l.strip() for l in txt.splitlines() if l.strip().startswith("#include")]
    has_torch = any("torch" in l for l in inc)
    has_at = any("ATen" in l or "c10" in l for l in inc)
    rows.append((rel, len(txt), has_torch or has_at, inc[:4]))

free = [r for r in rows if not r[2]]
need = [r for r in rows if r[2]]
print(f"torch-free: {len(free)}   need-torch: {len(need)}")
print()
print("=== torch-free (baseline-able as cubin) ===")
for rel, size, _, inc in sorted(free, key=lambda r: -r[1])[:18]:
    print(f"  {size/1024:6.1f} KB  {rel}")
