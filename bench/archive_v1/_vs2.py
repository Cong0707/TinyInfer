print("=== 对标 vLLM ===")
print()
print("[公开实测 RTX 5090, sm_120, 同代]")
print("  vLLM NVFP4 cutlass:  batch1 = 100 tok/s")
print("  AWQ int4:            batch1 = 150 tok/s")
print()
print("[vLLM.cpp 自研 fp4 MMA]")
print("  302 us per M=1 expert GEMM")
print()
print("[我们的 RTX 5060 Laptop, sm_120]")
print("  16384x5120 fp4 = 40 MB, gemv_fp4_fast: 613 us = 68 GB/s")
print()
H, MI = 5120, 2304
bytes_expert = H * MI * 0.5
print(f"一个真实专家 {H}x{MI} = {bytes_expert/2**20:.1f} MB fp4")
total_mb = 16384*5120*0.5/2**20
us_per_mb = 613 / total_mb
print(f"我们 {us_per_mb:.2f} us/MB -> 单个专家 {bytes_expert/2**20*us_per_mb:.0f} us")
ours = bytes_expert/2**20*us_per_mb
print()
print(f"vLLM.cpp 自研: 302 us")
print(f"我们:          {ours:.0f} us")
if ours < 302:
    print(f"-> 我们快 {302/ours:.2f}x")
else:
    print(f"-> 我们慢 {ours/302:.2f}x")