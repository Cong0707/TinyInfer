GB = 2**30; MB = 2**20
print("=== H200 跑 FP4 专家的反量化代价 ===")
print()
EXP4 = 18.38 * MB          # fp4 expert blob on disk
# fp4 -> bf16 is 4x the bytes (4 bits -> 16 bits)
EXP_BF16 = EXP4 * 4
print(f"一个专家: fp4 {EXP4/MB:.2f} MB -> bf16 反量化后 {EXP_BF16/MB:.2f} MB")
print(f"  fp4 -> fp8 (2x): {EXP4*2/MB:.2f} MB")
print()
TOPK = 6; L = 40
per_tok_fp4 = TOPK * L * EXP4 / GB
per_tok_bf16 = TOPK * L * EXP_BF16 / GB
per_tok_fp8 = TOPK * L * EXP4 * 2 / GB
print(f"每 token 240 个专家:")
print(f"  直接读 fp4 (Blackwell): {per_tok_fp4:.2f} GB")
print(f"  反量化到 fp8 (H200 折中): {per_tok_fp8:.2f} GB")
print(f"  反量化到 bf16: {per_tok_bf16:.2f} GB")
print()
print("=== 每 token 总读取量（含 dense + head）===")
DENSE = 6.84; HEAD = 1.26
for name, exp in (("Blackwell fp4", per_tok_fp4), ("H200 fp8", per_tok_fp8), ("H200 bf16", per_tok_bf16)):
    tot = DENSE + exp + HEAD
    print(f"  {name:16s}: {DENSE:.2f} + {exp:.2f} + {HEAD:.2f} = {tot:.2f} GB/token")
print()
print("=== 速度对比（3 卡）===")
BW = 3 * 4.8   # TB/s aggregate
for name, exp in (("Blackwell fp4", per_tok_fp4), ("H200 fp8", per_tok_fp8), ("H200 bf16", per_tok_bf16)):
    tot = DENSE + exp + HEAD
    ms = tot / BW * 1000
    print(f"  {name:16s}: {tot:.2f} GB / {BW:.1f} TB/s = {ms:.2f} ms -> {1000/ms:.0f} tok/s")