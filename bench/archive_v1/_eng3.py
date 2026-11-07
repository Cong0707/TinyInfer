import sys, time
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
import torch
import tinyinfer.torch_compat  # noqa: F401
from tinyinfer import gdn_patch, qwen_norm
gdn_patch.install(); qwen_norm.install()
from tinyinfer.engine import Engine
from transformers import AutoTokenizer
MD = r"E:\Models\Qwen3.5-0.8B"
tok = AutoTokenizer.from_pretrained(MD, trust_remote_code=True)
eng = Engine.from_hf(MD, device="cuda", allow_missing_layers=True)
eng._kv = eng.adapter.make_kv(64)
ids = tok("hi", return_tensors="pt").input_ids.to("cuda")
try:
    with torch.inference_mode():
        out = eng.forward(ids, 0)
    print("forward ok, logits", tuple(out.shape), "| top:", int(out[0, -1].argmax()), flush=True)
except Exception:
    import traceback; traceback.print_exc()
