
import torch, sys
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from tinyinfer.deepseek.hc import HCBlockMixin, _hc_pre_ref
from tinyinfer.nn import rms_norm
from transformers import AutoTokenizer

captured = []
orig_pre = HCBlockMixin.hc_pre
def patched(self, x, pre_mix):
    out = orig_pre(self, x, pre_mix)
    if len(captured) < 4:
        alt = _hc_pre_ref(x, pre_mix)
        d = (out.float() - alt.float()).abs().max().item()
        captured.append((self.layer_id, tuple(x.shape), tuple(pre_mix.shape),
                         tuple(out.shape), tuple(alt.shape), d))
    return out
HCBlockMixin.hc_pre = patched

d = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
ids = tok("Hello", return_tensors="pt").input_ids[0].tolist()
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device="cuda", layers=list(range(40)))
with torch.inference_mode():
    list(eng.stream_generate(ids, max_new_tokens=3, temperature=0.0))
HCBlockMixin.hc_pre = orig_pre
for lid, xs, ps, os_, as_, dd in captured:
    print(f"layer {lid:2d} x={xs} pre={ps}\n    native={os_}  ref={as_}  maxdiff={dd:.3e}")
