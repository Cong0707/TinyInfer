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
d = GraphDecoder.from_prompt(model, ids).capture()
# pure decode: argmax per step, no capture in the loop
with torch.inference_mode():
    for _ in range(10): d.next_token()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(128): d.next_token()
    dt = time.perf_counter() - t0
print(f"next_token loop: {128/dt:.1f} tok/s ({dt/128*1000:.2f}ms/tok)", flush=True)
# same but sampling-style: replay_step + argmax + set_token
d2 = GraphDecoder.from_prompt(model, ids).capture()
with torch.inference_mode():
    for _ in range(10): d2.replay_step(); d2.set_token(int(d2.logits_buf[0].argmax()))
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(128):
        d2.replay_step()
        d2.set_token(int(d2.logits_buf[0].argmax()))
    dt = time.perf_counter() - t0
print(f"replay_step+argmax: {128/dt:.1f} tok/s ({dt/128*1000:.2f}ms/tok)", flush=True)
