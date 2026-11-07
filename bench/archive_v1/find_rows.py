p = r"E:\Develop\Deepseek\tinyinfer\docs\kernels.md"
c=open(p,encoding="utf-8").read()
print("bs_bt", c.count(chr(92)+chr(96)), "chars", len(c))
print([l for l in c.splitlines() if "bench_hc_fused" in l or "fused/bench_gemv" in l or "fused/gemv_fp8" in l])
