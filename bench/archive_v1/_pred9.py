import sys, torch, traceback
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
eng.enable_prefetch = True
# 手工复现一次
b0 = eng.block(3); b1 = eng.block(4)
print(f'RESULT b1 has hc_mixes: {hasattr(b1, "hc_mixes")}')
print(f'RESULT b1 has ffn_norm: {hasattr(b1, "ffn_norm")}')
print(f'RESULT b1 ffn gate: {hasattr(b1.ffn, "gate")}')
x = torch.randn(1,1,4,5120, dtype=torch.bfloat16, device='cuda')
try:
    fp, fpo, fc = b1.hc_mixes(x, b1.hc_ffn_fn, b1.hc_ffn_scale, b1.hc_ffn_base)
    print(f'RESULT hc_mixes ok, fp {tuple(fp.shape)}')
    from tinyinfer.deepseek.hc import hc_pre_rms_norm
    gx = hc_pre_rms_norm(x, fp, b1.ffn_norm, eng.cfg.rms_norm_eps)
    print(f'RESULT gx {tuple(gx.shape)}')
    ids = b1.ffn.gate.forward(gx.reshape(-1, gx.shape[-1])[-1:].to(torch.bfloat16))[1]
    print(f'RESULT ids {ids.reshape(-1).tolist()}')
except Exception:
    traceback.print_exc()