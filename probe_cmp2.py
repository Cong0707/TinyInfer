"""对比：同样 layers=[0,1,2]，_forward_chunk 与手动逐层，h 是否一致。"""
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
M = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
t = "1+1 等于几？只回答数字。"
ids = torch.tensor([tok.encode(t, add_special_tokens=False)], device="cuda")
e = Engine(M, tier=OptLevel.AGGRESSIVE, device="cuda", layers=[0,1,2], tokenizer=tok)
print("gpu_layers:", e.gpu_layers, "stream:", sorted(e._stream_layers()))
print("layer_ids:", e.layer_ids, flush=True)

# 路径 A：_forward_chunk
e.reset()
la = e._forward_chunk(ids, 0)
torch.cuda.synchronize()

# 路径 B：手动
e.reset()
h = e.embed[ids.cpu()].to(e.device) if e.embed_on_cpu else e.embed[ids]
h = h.unsqueeze(2).repeat(1, 1, e.cfg.hc_mult, 1)
pm = torch.zeros(h.size(0), h.size(1), e.cfg.hc_mult, dtype=torch.float32, device=h.device)
pm[:, :, 0] = 1.0
hid = e.hashes(ids.reshape(-1).tolist(), 0) if e.hashes is not None else None
for i in (0, 1, 2):
    b = e.block(i)
    if getattr(b, "engram", None) is not None and hid is not None:
        h = b.engram.forward(h, hid[b.engram.hash_index])
    h, pm = b.forward(h, 0, pm)
    print("  manual layer %d out absmax=%.3f" % (i, float(h.float().abs().max())), flush=True)

print("A finite=%s" % bool(torch.isfinite(la).all()), flush=True)
