import sys
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
import torch
import tinyinfer.torch_compat  # noqa: F401
from tinyinfer import gdn_patch, qwen_norm
gdn_patch.install(); qwen_norm.install()
from tinyinfer.engine import Engine
from transformers import AutoModelForCausalLM, AutoTokenizer

MD = r"E:\Models\Qwen3.5-0.8B"
tok = AutoTokenizer.from_pretrained(MD, trust_remote_code=True)
ids = tok("hi", return_tensors="pt").input_ids.to("cuda")

ref = AutoModelForCausalLM.from_pretrained(MD, dtype=torch.bfloat16,
                                           trust_remote_code=True).eval().to("cuda")
# capture the hidden state after layer 0 from the reference
hidden = {}
def hook(mod, inp, out):
    if isinstance(out, tuple):
        hidden["l0"] = out[0].detach()
    else:
        hidden["l0"] = out.detach()
h = ref.model.layers[0].register_forward_hook(hook)
with torch.inference_mode():
    ref(input_ids=ids)
h.remove()
print("ref layer0 hidden:", tuple(hidden["l0"].shape), "amax", float(hidden["l0"].abs().max()), flush=True)
del ref
torch.cuda.empty_cache()

eng = Engine.from_hf(MD, device="cuda", allow_missing_layers=True)
kv = eng.adapter.make_kv(64)
eng._kv = kv
with torch.inference_mode():
    x = torch.nn.functional.embedding(ids, eng.embed)
    pos = torch.arange(0, ids.shape[1], device=ids.device)
    b0 = eng.block(0)
    y = b0.forward(x, cache=kv, position=pos)
print("eng layer0 hidden:", tuple(y.shape), "amax", float(y.abs().max()), flush=True)
d = (y[0,-1].float() - hidden["l0"][0,-1].float()).abs()
print("diff amax:", float(d.max()), "| mean:", float(d.mean()), flush=True)
