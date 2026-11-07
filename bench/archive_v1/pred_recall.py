import sys, time, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from tinyinfer.routes import RoutePredictor, RoutePredictorConfig

torch.manual_seed(0)
eng = Engine(r"E:\Develop\Deepseek\tinyinfer\cache\weights", tier=OptLevel.BALANCED,
             device="cuda", layers=list(range(5)))
eng.enable_prefetch = False

prompt = [19923, 14, 342, 1030, 22651, 4374, 1465, 16, 17, 18, 19, 20, 21, 22, 23, 24]
with torch.inference_mode():
    logits = eng.forward(torch.tensor([prompt]), 0)
    nid = int(logits[0, -1].argmax())
    N = 40
    for width in (6, 8, 12, 16, 24):
        pred = RoutePredictor(RoutePredictorConfig(width=width))
        # replay the same token stream to keep the comparison fair
        torch.manual_seed(0)
        eng.reset()
        logits = eng.forward(torch.tensor([prompt]), 0)
        nid = int(logits[0, -1].argmax())
        cov = tot_actual = 0
        for step in range(N):
            p = len(prompt) + step
            pp = pred.predict(nid)
            logits = eng.forward(torch.tensor([[nid]]), p)
            cur = dict(eng._last_routing)
            pred.observe(nid, cur)
            for layer, actual in cur.items():
                plist = pp.get(layer, [])
                cov += len(set(plist) & set(actual))
                tot_actual += len(actual)
            nid = int(logits[0, -1].argmax())
        print(f"width={width:2d}: recall={cov/max(tot_actual,1):.3f}  pred={pred.stats()}")
