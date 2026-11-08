"""逐层核对压缩 KV 的字节数，与之前的 6.91GB 对比。"""
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
import json
GB = 2**30
cfg = json.load(open(r"E:\models\DeepSeek-V4.1-Flash-1017\config.json", encoding="utf-8"))["text_config"]
ratios = cfg["compress_ratios"][:40]
head_dim = cfg["head_dim"]
CTX = 1_048_576

print("%-5s %-6s %-10s %-12s" % ("layer", "ratio", "groups", "MB"), flush=True)
total = 0
for i, r in enumerate(ratios):
    if r <= 0:
        continue
    g = CTX // r
    b = g * head_dim
    total += b
    if i < 8 or i in (19, 20, 39):
        print("%-5d %-6d %-10d %-12.1f" % (i, r, g, b / 2**20), flush=True)
print("... 共 %d 个压缩层" % sum(1 for r in ratios if r > 0), flush=True)
print("total = %.2f GB" % (total / GB), flush=True)
print()
# 与 probe_kv.py 的口径对比：那个用的是 kv_heads=1 还是别的？
print("如果每 token 存 n_kv_heads x head_dim:", flush=True)
nkv = cfg.get("num_key_value_heads", 1)
print("  num_key_value_heads =", nkv, flush=True)
t2 = 0
for i, r in enumerate(ratios):
    if r <= 0:
        continue
    t2 += (CTX // r) * head_dim * nkv
print("total(nkv=%d) = %.2f GB" % (nkv, t2 / GB), flush=True)
