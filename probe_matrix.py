import sys, time, os, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer

MODEL = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

import gc
for n in (12, 40):
    for slots in (14, 32, 64):
        os.environ["TI_SLOTS"] = str(slots)
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            e = Engine(MODEL, tier=OptLevel.AGGRESSIVE, device="cuda",
                       layers=list(range(n)), tokenizer=tok)
        except Exception as exc:
            print("n=%d slots=%d  LOAD FAIL %s" % (n, slots, repr(exc)[:80]), flush=True)
            continue
        b = e.block(0)
        D, HC = e.cfg.dim, e.cfg.hc_mult
        x4 = torch.randn(1, 1, HC, D, dtype=torch.bfloat16, device="cuda") * 0.02
        pm = torch.zeros(1, 1, HC, dtype=torch.float32, device="cuda"); pm[:, :, 0] = 1.0
        posv = torch.tensor([0], device="cuda")
        try:
            for _ in range(3):
                b.forward(x4, 0, pm, positions=posv)
            torch.cuda.synchronize()
            e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            for _ in range(5):
                b.forward(x4, 0, pm, positions=posv)
            e1.record(); torch.cuda.synchronize()
            ms = e0.elapsed_time(e1) / 5
            print("n=%d slots=%d  layer=%.1fms  ->  x%d = %.0f ms/token" %
                  (n, slots, ms, n, ms * n), flush=True)
        except Exception as exc:
            print("n=%d slots=%d  RUN FAIL %s" % (n, slots, repr(exc)[:80]), flush=True)
        # 引擎有机器级锁，必须显式释放，否则下一次构造直接 EngineLockBusy
        try:
            e._engine_lock.release()
        except Exception:
            pass
        del e
        gc.collect()
        torch.cuda.empty_cache()
