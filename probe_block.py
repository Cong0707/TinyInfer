import sys, time, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer

MODEL = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
e = Engine(MODEL, tier=OptLevel.AGGRESSIVE, device="cuda", layers=[0, 1], tokenizer=tok)
b = e._blocks[0]
D = e.cfg.dim
HC = e.cfg.hc_mult

def bench(fn, n=20, warm=5):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e3

def _reset():
    for sub in (b.attn, getattr(b.attn, "compressor", None), getattr(b.attn, "indexer", None)):
        if sub is not None and hasattr(sub, "reset"):
            sub.reset()

pos = torch.tensor([0], device="cuda")
# block 输入：[b, s, hc_mult, dim]（hc_mixes 里 flatten(2) 印证）
x4 = torch.randn(1, 1, HC, D, dtype=torch.bfloat16, device="cuda") * 0.02
# pre_mix 是 [b, s, hc]（在 hc 轴上做加权，不是每维一个权重）
pm = torch.zeros(1, 1, HC, dtype=torch.float32, device="cuda")
pm[:, :, 0] = 1.0

_reset()
t_block = bench(lambda: b.forward(x4, 0, pm, positions=pos))
t_hc = bench(lambda: b.hc_mixes(x4, b.hc_attn_fn, b.hc_attn_scale, b.hc_attn_base))
t_pre = bench(lambda: b.hc_pre(x4, pm))

print("BLOCK_MS %.3f  HC_MIXES_MS %.3f  HC_PRE_MS %.3f" % (t_block, t_hc, t_pre), flush=True)
print("x40 block=%.0f ms/token   hc_mixes(2x)=%.0f   hc_pre(2x)=%.0f" %
      (t_block * 40, t_hc * 2 * 40, t_pre * 2 * 40), flush=True)
