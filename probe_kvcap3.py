"""压缩 KV 的真实占用：latent 是 fp4（0.5 字节/元素）。"""
import sys, json
GB = 2**30
cfg = json.load(open(r"E:\models\DeepSeek-V4.1-Flash-1017\config.json", encoding="utf-8"))["text_config"]
ratios = cfg["compress_ratios"][:40]
head_dim = cfg["head_dim"]
CTX = 1_048_576

for label, bpe in (("fp8 (1.0 B)", 1.0), ("fp4 (0.5 B)", 0.5)):
    total = sum((CTX // r) * head_dim * bpe for r in ratios if r > 0)
    print("%-12s 1M -> %.2f GB" % (label, total / GB), flush=True)

print()
print("代码注释里的实测值: 6.92 GB", flush=True)
print("fp4 计算值:        %.2f GB  <- 吻合" % (
      sum((CTX // r) * head_dim * 0.5 for r in ratios if r > 0) / GB), flush=True)
print()
# L20 上能跑几条 1M 会话
per_session = sum((CTX // r) * head_dim * 0.5 for r in ratios if r > 0)
print("L20 (44GB，扣掉层权重/缓冲约 8GB -> 36GB 可用):", flush=True)
for avail in (36, 40):
    print("  可用 %dGB -> %d 条 1M 会话" % (avail, int(avail * GB // per_session)), flush=True)
