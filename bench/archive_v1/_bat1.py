
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer

d = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device="cuda", layers=list(range(2)))
a = tok("Hello world", return_tensors="pt").input_ids[0].tolist()
b = tok("The quick brown", return_tensors="pt").input_ids[0].tolist()
print("len a", len(a), "len b", len(b))
S = max(len(a), len(b))
pa = a + [tok.eos_token_id or 0] * (S - len(a))
pb = b + [tok.eos_token_id or 0] * (S - len(b))
with torch.inference_mode():
    # 单序列基线
    eng.reset()
    l1 = eng.forward(torch.tensor([a], device="cuda"), 0)
    print("M=1 logits", tuple(l1.shape))
    # 批 2
    eng.reset()
    try:
        l2 = eng.forward(torch.tensor([pa, pb], device="cuda"), 0)
        torch.cuda.synchronize()
        print("M=2 logits", tuple(l2.shape))
        same = (l2[0, len(a)-1].float() - l1[0, -1].float()).abs().max().item()
        print("a 行与单序列对比 maxdiff:", same)
    except Exception as e:
        import traceback; traceback.print_exc()
