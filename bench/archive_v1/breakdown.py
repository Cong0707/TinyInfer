import sys, time, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel

torch.manual_seed(0)
eng = Engine(r"E:\Develop\Deepseek\tinyinfer\cache\weights", tier=OptLevel.BALANCED,
             device="cuda", layers=[0,1,2,3,4,5,6])
prompt = [19923, 14, 342, 1030, 22651, 4374, 1465, 16, 17, 18, 19, 20, 21, 22, 23, 24]

def stats(fn, n=3):
    for _ in range(2): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e3

with torch.inference_mode():
    # warmup
    logits = eng.forward(torch.tensor([prompt]), 0)
    nid = int(logits[0, -1].argmax())
    eng.forward(torch.tensor([[nid]]), 16)

    ids = torch.tensor([[nid]], device="cuda")
    pos = 17
    # full token
    with torch.inference_mode():
        full = stats(lambda: eng.forward(ids, pos))
        pos += 0
    print(f"token total: {full:.1f} ms")

    # embed + first-layer setup
    h = stats(lambda: eng.embed[ids])
    print(f"embed: {h:.2f} ms")

    # per-layer: use cached blocks
    for i in eng.layer_ids:
        b = eng.block(i)
        x = torch.randn(1, 1, 4, 5120, dtype=torch.bfloat16, device="cuda") * 0.02
        pre = torch.zeros(1, 1, 4, dtype=torch.float32, device="cuda"); pre[:, :, 0] = 1
        b.attn.reset(); b.ffn._expert_cache.clear()
        t_whole = stats(lambda: b.forward(x, 5, pre))
        t_attn = stats(lambda: b.attn.forward(x[:, :, 0, :], 5))
        t_ffn = stats(lambda: b.ffn.forward(x[:, :, 0, :]))
        print(f"L{i}: block {t_whole:5.1f}  attn {t_attn:5.1f}  moe {t_ffn:5.1f}  hc {t_whole-t_attn-t_ffn:5.1f}")
