import sys, time, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer

torch.manual_seed(0)
d = r"E:\Develop\Deepseek\tinyinfer\cache\weights"
tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
t0 = time.time()
eng = Engine(d, tier=OptLevel.BALANCED, device="cuda", layers=list(range(40)))
print(f"load: {time.time()-t0:.1f}s layers={len(eng.layer_ids)}", flush=True)
prompt = tok("Hello, I am DeepSeek.", return_tensors="pt").input_ids[0].tolist()
with torch.inference_mode():
    # prefill (warmup incl. JIT)
    t0 = time.time(); logits = eng.forward(torch.tensor([prompt]), 0); torch.cuda.synchronize()
    print(f"prefill {len(prompt)} tok (cold): {time.time()-t0:.1f}s", flush=True)
    # decode
    nid = int(logits[0, -1].argmax()); pos = len(prompt)
    t0 = time.time()
    for _ in range(4):
        logits = eng.forward(torch.tensor([[nid]]), pos); pos += 1
        nid = int(logits[0, -1].argmax())
    torch.cuda.synchronize()
    dt = (time.time()-t0)/4
    print(f"decode: {dt*1000:.0f} ms/token ({1/dt:.2f} tok/s)", flush=True)
    # generation e2e
    eng.reset()
    t0 = time.time()
    outs = list(eng.stream_generate(prompt, max_new_tokens=8, temperature=0.0))
    print(f"gen: {tok.decode(outs)!r} in {time.time()-t0:.1f}s", flush=True)
print("stats:", {k: v for k, v in eng.stats().items() if k in ("layers_on_gpu", "vram_peak_mb")}, flush=True)
