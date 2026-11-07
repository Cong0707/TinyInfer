import sys, time, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel

eng = Engine(r"E:\Develop\Deepseek\tinyinfer\cache\weights", tier=OptLevel.BALANCED,
             device="cuda", layers=list(range(26)))
P = {"predicts": 0, "correct": 0, "total_experts": 0}
if eng.predictor is not None:
    orig = eng.predictor.predict
    def pred(tok):
        r = orig(tok)
        if r:
            P["predicts"] += 1
            P["total_experts"] += sum(len(v) for v in r.values())
        return r
    eng.predictor.predict = pred
with torch.inference_mode():
    eng.forward(torch.tensor([[1, 2, 3, 4]]), 0)
    nid = int(eng.forward(torch.tensor([[1]]), 4)[0, -1].argmax())
    P["predicts"] = 0
    P["total_experts"] = 0
    t0 = time.time()
    logits = eng.forward(torch.tensor([[nid]]), 5)
    torch.cuda.synchronize()
    wall = time.time() - t0
print(f"wall {wall*1000:.0f}ms | predictor calls={P['predicts']} predicted_experts={P['total_experts']}")
