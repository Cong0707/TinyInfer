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

# reference
ref = AutoModelForCausalLM.from_pretrained(MD, dtype=torch.bfloat16,
                                           trust_remote_code=True).eval().to("cuda")
with torch.inference_mode():
    r = ref(input_ids=ids).logits
print("reference top:", int(r[0, -1].argmax()), flush=True)
del ref
torch.cuda.empty_cache()

eng = Engine.from_hf(MD, device="cuda", allow_missing_layers=True)
eng._kv = eng.adapter.make_kv(64)
with torch.inference_mode():
    o = eng.forward(ids, 0)
print("engine    top:", int(o[0, -1].argmax()), flush=True)
k = min(5, o.shape[-1])
print("engine  top5:", o[0, -1].topk(5).indices.tolist(), flush=True)
print("engine  amax:", float(o.abs().max()), "| ref amax:", float(r.abs().max()), flush=True)
