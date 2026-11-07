import sys, time
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
import torch
import tinyinfer.torch_compat  # noqa: F401
from tinyinfer import gdn_patch, qwen_norm
gdn_patch.install(); qwen_norm.install()
from tinyinfer.engine import Engine
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F

MD = r"E:\Models\Qwen3.5-0.8B"
tok = AutoTokenizer.from_pretrained(MD, trust_remote_code=True)
ids = tok("hi", return_tensors="pt").input_ids.to("cuda")
ref = AutoModelForCausalLM.from_pretrained(MD, dtype=torch.bfloat16,
                                           trust_remote_code=True).eval().to("cuda")
with torch.inference_mode():
    rl = ref(input_ids=ids).logits
print("reference top5:", rl[0, -1].topk(5).indices.tolist(), flush=True)
del ref
torch.cuda.empty_cache()

eng = Engine.from_hf(MD, device="cuda", allow_missing_layers=True)
kv = eng.adapter.make_kv(64)
model = eng.adapter.model
with torch.inference_mode():
    pos = torch.arange(0, ids.shape[1], device="cuda")
    h = F.embedding(ids, eng.embed)
    rope = model.position_embeddings(h, pos)
    tp = model.text_positions(pos)
    for i in range(eng.cfg.n_layers):
        lay = eng.block(i)
        mask = model.attention_mask(h, tp, cache=kv, layer_id=i)
        out = lay.forward(h, cache=kv, position=tp, position_embeddings=rope,
                          attention_mask=mask)
        h = out[0] if isinstance(out, tuple) else out
    hn = model.final_norm(h[:, -1:])
    lg = F.linear(hn, eng.head)
print("engine    top5:", lg[0, -1].topk(5).indices.tolist(), flush=True)
d = (lg.float() - rl.float()).abs()
print("logits diff amax:", float(d.max()), "| mean:", float(d.mean()), flush=True)
