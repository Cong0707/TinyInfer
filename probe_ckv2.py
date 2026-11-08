"""压缩组的初始化状态：未写入的组里是什么。"""
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
M = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
base = "1+1 等于几？只回答数字。请仔细思考后再回答，不要着急。再加一些字凑够。 "
enc = tok.encode(base, add_special_tokens=False)

import tinyinfer.deepseek.compressed_attention as CA
orig = CA.MLACompressedAttention.forward
rec = []
def hooked(self, x, start_pos=0, positions=None):
    out = orig(self, x, start_pos, positions)
    torch.cuda.synchronize()
    rt = getattr(self, "rt", None)
    ck = getattr(rt, "compress_kv", None) if rt is not None else None
    if torch.is_tensor(ck):
        # 逐组检查有多少组是「全零」（未写入）
        g = ck[0].float()
        nz = (g.abs().sum(-1) > 0).sum().item()
        bad = int(torch.isnan(g).sum())
        rec.append((getattr(self, "layer_id", "?"), ck.shape[1], int(nz), bad,
                    bool(torch.isfinite(g).all())))
    return out
CA.MLACompressedAttention.forward = hooked

N = int(sys.argv[1])
ids = torch.tensor([enc[:N]], device="cuda")
e = Engine(M, tier=OptLevel.AGGRESSIVE, device="cuda", layers=[0,1,2,3,4], tokenizer=tok)
e.reset()
lg = e._forward_chunk(ids, 0)
torch.cuda.synchronize()
f = lg.float()
bad = torch.isnan(f).any(dim=-1)[0].nonzero().flatten().tolist()
print("N=%d nan_positions=%s compress_len=%d" % (N, bad, N // 2), flush=True)
for lid, cap, nz, nan, ok in rec:
    print("  L%s cap=%d written_groups=%d nan_elems=%d all_finite=%s" %
          (lid, cap, nz, nan, ok), flush=True)
