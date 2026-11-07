import sys, time
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
text = tok.apply_chat_template([{"role":"user","content":"hi"}], tokenize=False,
                               add_generation_prompt=True)
ids = tok(text, return_tensors="pt").input_ids.to("cuda")
for _ in range(3):
    t0 = time.perf_counter()
    d = GraphDecoder.from_prompt(model, ids).capture()
    torch.cuda.synchronize()
    print(f"capture: {(time.perf_counter()-t0)*1000:.0f}ms  graph={d.using_graph} "
          f"prerolls={len(getattr(d,'pending',[]) or [])}", flush=True)
    del d; torch.cuda.empty_cache()
