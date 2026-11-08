"""4 层（含压缩层）：单次 2048 vs 分块 2048，cold vs warm。"""
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
M = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
e = Engine(M, tier=OptLevel.AGGRESSIVE, device="cuda", layers=[0,1,2,3], tokenizer=tok)
base = tok.encode("你好今天天气不错", add_special_tokens=False)
N = 2048
ids = torch.tensor([(base*2000)[:N]], device="cuda")
STEP = int(sys.argv[1])   # 2048 = 单次
WARM = sys.argv[2] == "1"

if WARM:
    for w in (64, 256, 512, 1024):
        e.reset()
        e._forward_chunk(torch.tensor([(base*2000)[:w]], device="cuda"), 0)
    torch.cuda.synchronize()

e.reset()
bad = -1
try:
    for off in range(0, N, STEP):
        lg = e._forward_chunk(ids[:, off:off+STEP], off)
        torch.cuda.synchronize()
        if not bool(torch.isfinite(lg).all()):
            bad = off
            break
except Exception as ex:
    print("RESULT step=%d warm=%s EXC %s" % (STEP, WARM, str(ex)[:60]), flush=True)
    sys.exit(0)
print("RESULT step=%d warm=%s bad_chunk=%d" % (STEP, WARM, bad), flush=True)
