
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer import vllm_ops as V

# 静态追踪：给所有 vllm_ops 函数打桩，看引擎调用谁
call_log = {}
_orig = {}
for name in ("rms_norm", "silu_and_mul", "clamp_swiglu", "gate_topk"):
    fn = getattr(V, name)
    def make(n, f):
        def wrapper(*a, **k):
            r = f(*a, **k)
            call_log[n] = call_log.get(n, 0) + 1
            return r
        return wrapper
    setattr(V, name, make(name, fn))

# gemv_fp4_marlin 也打桩
from tinyinfer import gemv_fp4_marlin as GM
_orig_fast = GM.gemv_fp4_fast
def fast(*a, **k):
    call_log["gemv_fp4_fast"] = call_log.get("gemv_fp4_fast", 0) + 1
    return _orig_fast(*a, **k)
GM.gemv_fp4_fast = fast

from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
import tinyinfer.moe as MOE
# moe.py 是 from .vllm_ops import gate_topk 这种直接绑定，需要同步替换
MOE.gate_topk = V.gate_topk
import tinyinfer.nn as NN
if hasattr(NN, "rms_norm"): NN.rms_norm = V.rms_norm
import tinyinfer.gemm as GEMM
# gemm 里是 from .gemv_fp4_marlin import gemv_fp4_fast as _marlin_gemv（函数内 import）
d = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
ids = tok("Hello, I am DeepSeek.", return_tensors="pt").input_ids[0].tolist()
eng = Engine(d, tier=OptLevel.BALANCED, device="cuda", layers=list(range(6)))
call_log.clear()
with torch.inference_mode():
    list(eng.stream_generate(ids, max_new_tokens=2, temperature=0.0))
print("=== 6 层 2 token 的 vLLM 算子调用次数 ===")
for k, v in sorted(call_log.items(), key=lambda x: -x[1]):
    print(f"  {k:<22} {v}")
if not call_log:
    print("  (无)")
