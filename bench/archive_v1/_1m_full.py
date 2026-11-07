print("=== 1M 上下文的三个成本 ===")
print()
ACTIVE = 37e9
CTX = 1048576

print("1) PREFILL 一条 1M 上下文（一次性）")
print(f"   FLOPs = 2 x {ACTIVE/1e9:.0f}B x {CTX:,} = {2*ACTIVE*CTX/1e15:.1f} PFLOP")
for n, fp8 in ((4, 4*1979e12), (8, 8*1979e12)):
    sec = 2*ACTIVE*CTX / fp8
    print(f"   {n} x H200 fp8 ({fp8/1e12:.0f} TFLOP/s, 100% 效率): {sec:.0f} s")
    print(f"     ~50% 实际效率: {sec*2:.0f} s；算上 attention 的 O(n^2) 部分再翻倍")
print()
print("   注意：1M 的 attention 是 O(n^2)。即使 MLA 压缩了 KV，")
print("   query 仍是 1M 个 token，attention 的 FLOPs 远超线性项。")
q_heads = 64; hd = 512
attn_flops = 2 * CTX * CTX * hd * 2  # q@k 和 attn@v
print(f"   attention q@k^T: 2 x {CTX}^2 x {hd} = {attn_flops/1e15:.0f} PFLOP")
print(f"   -> 比线性部分大 {attn_flops/(2*ACTIVE*CTX):.0f} 倍")
print()
print("2) DECODE 时每条序列每 token 要扫全部 1M KV")
kv = 1.25 * 2**30   # fp8, from the KV calc
print(f"   KV 1.25 GB/seq，每条序列每 token 读一遍")
for n, bw in ((4, 19.2e12), (8, 38.4e12)):
    ms = kv / bw * 1000
    print(f"   {n} 卡 {bw/1e12:.1f} TB/s: {ms:.2f} ms/seq/token")
print()
print("3) 并发的真实上限")
print("   显存说 222 条（4卡fp8），但 decode 时每条都要读自己的 1.25GB KV，")
print("   带宽必须服务所有并发序列：")
for n, bw in ((4, 19.2e12), (8, 38.4e12)):
    print(f"   {n} 卡 {bw/1e12:.1f} TB/s / 1.25 GB = 每秒 {bw/kv:.0f} 条序列·token")
    for conc in (10, 50, 222):
        tps = bw / (kv * conc)
        print(f"     并发 {conc:3d} 条: 每条 {tps:.1f} tok/s, 合计 {tps*conc:.0f} tok/s")
    print()