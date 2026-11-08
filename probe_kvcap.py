"""压缩 KV 的真实容量规划：按 1M 上下文算，与几何增长核对。"""
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
GB = 2**30

# config 里的真实值
import json
cfg = json.load(open(r"E:\models\DeepSeek-V4.1-Flash-1017\config.json", encoding="utf-8"))["text_config"]
ratios = cfg["compress_ratios"][:40]
head_dim = cfg["head_dim"]

print("compress_ratios 前 40:", ratios, flush=True)
print("head_dim:", head_dim, flush=True)

CTX = 1_048_576
total = 0
for i, r in enumerate(ratios):
    if r <= 0:
        continue
    groups = CTX // r
    per_layer = groups * head_dim      # fp8 = 1 byte
    total += per_layer
print("1M 上下文：压缩 KV 总量 = %.2f GB" % (total / GB), flush=True)

# 几何增长的峰值（每次翻倍，最后一次可能超过需求）
peak = 0
for i, r in enumerate(ratios):
    if r <= 0:
        continue
    need = CTX // r
    cap = 1024
    while cap < need:
        cap *= 2
    peak += cap * head_dim
print("几何增长（按 2 的幂）峰值 = %.2f GB" % (peak / GB), flush=True)
print("浪费 = %.2f GB" % ((peak - total) / GB), flush=True)
