
import torch, sys
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
d = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device="cuda", layers=list(range(2)))
b = eng.block(0)
print("attn_norm:", b.attn_norm.dtype, tuple(b.attn_norm.shape), type(b.attn_norm).__name__)
print("ffn_norm :", b.ffn_norm.dtype, tuple(b.ffn_norm.shape))
print("hc_mult  :", b.hc_mult, "eps:", eng.cfg.rms_norm_eps)
x = torch.randn(1,1,b.hc_mult,8, dtype=torch.bfloat16, device="cuda")
print("x:", tuple(x.shape), x.dtype)
