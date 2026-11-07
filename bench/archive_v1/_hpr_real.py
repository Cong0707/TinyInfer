
import torch, sys
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from tinyinfer.deepseek.hc import hc_pre_rms_norm
from tinyinfer.nn import rms_norm
from transformers import AutoTokenizer
d = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device="cuda", layers=list(range(2)))
b = eng.block(0)
torch.manual_seed(0)
x = torch.randn(1, 1, b.hc_mult, 5120, dtype=torch.bfloat16, device="cuda") * 0.1
pre, post, comb = b.hc_mixes(x, b.hc_attn_fn, b.hc_attn_scale, b.hc_attn_base)
print("pre_mix:", tuple(pre.shape), pre.dtype)

h_split = rms_norm(b.hc_pre(x, pre), b.attn_norm, eng.cfg.rms_norm_eps)
h_fused = hc_pre_rms_norm(x, pre, b.attn_norm, eng.cfg.rms_norm_eps)
torch.cuda.synchronize()
print("split:", tuple(h_split.shape), "fused:", tuple(h_fused.shape))
diff = (h_split.float()-h_fused.float()).abs()
print(f"maxdiff={diff.max().item():.3e}  rel={diff.max().item()/h_split.float().abs().max().item():.2e}")
print("bit-identical:", bool((h_split==h_fused).all()))
