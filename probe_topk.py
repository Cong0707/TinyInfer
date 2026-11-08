"""top-k 从 6 降到 4 的代价与收益（8GB 卡上的真实权衡）。"""
import sys, os, time, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from tinyinfer.deepseek.model import TinyModel
from tinyinfer.config import TinyConfig
from transformers import AutoTokenizer

MODEL = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
cfg = TinyConfig.from_hf(MODEL)
print("checkpoint topk =", cfg.n_activated_experts, flush=True)

# 只测一层 MoE 的耗时与专家数，不跑全模型（8GB 装不下）
for topk in (6, 4):
    e = Engine(MODEL, tier=OptLevel.AGGRESSIVE, device="cuda", layers=[0], tokenizer=tok)
    b = e._blocks[0]
    if topk != 6:
        b.ffn.gate.topk = topk
    HC, D = cfg.hc_mult, cfg.dim
    h = torch.randn(1, 1, HC, D, dtype=torch.bfloat16, device="cuda") * 0.02
    from tinyinfer.nn import rms_norm
    pm = torch.zeros(1, 1, HC, dtype=torch.float32, device="cuda"); pm[:, :, 0] = 1.0
    x = rms_norm(b.hc_pre(h, pm), b.ffn_norm, cfg.rms_norm_eps)
    for _ in range(3):
        b.ffn.forward(x)
    torch.cuda.synchronize()
    e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
    e0.record()
    n = 10
    for _ in range(n):
        b.ffn.forward(x)
    e1.record(); torch.cuda.synchronize()
    ms = e0.elapsed_time(e1) / n
    ids = len(b.ffn._last_ids or [])
    print("topk=%d  moe=%.2fms  experts/layer=%d  -> x40 = %.0fms" %
          (topk, ms, ids, ms * 40), flush=True)
    try:
        e._engine_lock.release()
    except Exception:
        pass
    del e
    import gc; gc.collect(); torch.cuda.empty_cache()
