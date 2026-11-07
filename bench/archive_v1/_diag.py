import sys
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
import torch
import tinyinfer.torch_compat  # noqa: F401
from tinyinfer import gdn_patch, qwen_elt, qwen_norm
gdn_patch.install(); qwen_norm.install()
from transformers import AutoModelForCausalLM, AutoTokenizer
from tinyinfer.hf_graph import GraphDecoder
MD = r"E:\Models\Qwen3.5-0.8B"
tok = AutoTokenizer.from_pretrained(MD, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MD, dtype=torch.bfloat16,
                                             trust_remote_code=True).eval().to("cuda")
qwen_elt.install(model)
text = tok.apply_chat_template([{"role":"user","content":"1+1等于几？"}], tokenize=False,
                               add_generation_prompt=True)
ids = tok(text, return_tensors="pt").input_ids.to("cuda")
L = int(ids.shape[1])
print("prompt tokens:", L, flush=True)
d = GraphDecoder.from_prompt(model, ids).capture()
with torch.inference_mode():
    print("after capture: pos=", int(d.pos_buf[0]),
          "ready_pos=", getattr(d, "ready_pos", None),
          "ready_token=", getattr(d, "ready_token", None),
          "capture_token=", getattr(d, "capture_token", None),
          "pending=", getattr(d, "pending", None), flush=True)
    drain = d.drain()
    print("drain ->", drain, flush=True)
    first = [d.next_token() for _ in range(4)]
    print("replays after drain:", first, flush=True)
    # now rewind and see what the reused path would produce
    d.rewind()
    print("after rewind: pos=", int(d.pos_buf[0]), "ids=", int(d.ids_buf[0,0]), flush=True)
    reused = [d.next_token() for _ in range(4)]
    print("replays after rewind:", reused, flush=True)
