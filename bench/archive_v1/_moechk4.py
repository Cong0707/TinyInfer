
import torch, traceback
for mod in ("tinyinfer._moe_C_stable_libtorch", "vllm._moe_C_stable_libtorch"):
    try:
        m = __import__(mod, fromlist=["x"])
        ops = sorted(x for x in dir(torch.ops._moe_C) if not x.startswith("_"))
        print(f"{mod}: imported, _moe_C ops = {len(ops)} {ops[:8]}")
    except Exception as exc:
        print(f"{mod}: FAIL {type(exc).__name__}: {exc}")
print("done")
