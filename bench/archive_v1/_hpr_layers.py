
import torch, sys
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from tinyinfer.deepseek.hc import hc_pre_rms_norm
from tinyinfer.nn import rms_norm
from transformers import AutoTokenizer
d = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
ids = tok("Hello", return_tensors="pt").input_ids[0].tolist()
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device="cuda", layers=list(range(40)))
with torch.inference_mode():
    list(eng.stream_generate(ids, max_new_tokens=3, temperature=0.0))
    # 取一个流式层和一个常驻层分别测
    for lid in (0, 35):
        b = eng.block(lid)
        torch.manual_seed(0)
        x = torch.randn(1, 1, b.hc_mult, 5120, dtype=torch.bfloat16, device="cuda") * 0.1
        try:
            pre, _, _ = b.hc_mixes(x, b.hc_attn_fn, b.hc_attn_scale, b.hc_attn_base)
            hs = rms_norm(b.hc_pre(x, pre), b.attn_norm, eng.cfg.rms_norm_eps)
            hf = hc_pre_rms_norm(x, pre, b.attn_norm, eng.cfg.rms_norm_eps)
            torch.cuda.synchronize()
            dm = (hs.float()-hf.float()).abs().max().item()
            print(f"layer {lid:2d}: shape {tuple(hf.shape)} maxdiff={dm:.3e} bit={bool((hs==hf).all())} "
                  f"norm_dtype={b.attn_norm.dtype} contig={b.attn_norm.is_contiguous()}")
        except Exception as e:
            print(f"layer {lid}: FAILED {type(e).__name__} {str(e)[:110]}")
