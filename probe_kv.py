"""1M 上下文的 KV 实际占多少显存 —— 按模型真实结构算，不猜。"""
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.config import TinyConfig
from transformers import AutoTokenizer

MODEL = r"E:\models\DeepSeek-V4.1-Flash-1017"
cfg = TinyConfig.from_hf(MODEL)
print("layers=%d dim=%d head_dim=%d window=%d" %
      (cfg.n_layers, cfg.dim, cfg.head_dim, cfg.sliding_window), flush=True)
print("compress_ratios=%s" % (list(cfg.compress_ratios)[:8],), flush=True)

for N in (128_000, 1_000_000):
    win_bytes = 0
    comp_bytes = 0
    for lid in range(cfg.n_layers):
        # window：固定 128 格，bf16
        win_bytes += cfg.sliding_window * cfg.head_dim * 2
        # 压缩：N/ratio 组，每组 head_dim（latent 是 fp4 存的，按 2 值/字节）
        r = cfg.compress_ratios[lid]
        if r > 0:
            ncomp = N // r
            comp_bytes += ncomp * cfg.head_dim // 2
    total = win_bytes + comp_bytes
    print("N=%7d  window=%.2f GB  compressed=%.2f GB  total=%.2f GB" %
          (N, win_bytes / 2**30, comp_bytes / 2**30, total / 2**30), flush=True)
