"""逐层比较 batch 与逐 token 的 hidden，找第一处不同。"""
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
M = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
e = Engine(M, tier=OptLevel.AGGRESSIVE, device="cuda", layers=[0,1,2,3], tokenizer=tok)
base = tok.encode("1+1 等于几？只回答数字。请仔细思考。", add_special_tokens=False)
N = 8
ids = torch.tensor([base[:N]], device="cuda")

# batch 路径，记录每层输出
e.reset()
e._dump_h = []
e._forward_chunk(ids, 0)
torch.cuda.synchronize()
batch_layers = [(i, hin.clone(), hout.clone()) for i, hin, hout in e._dump_h]
e._dump_h = None

# token 路径，记录每层在最后一个位置的输出
import tinyinfer.deepseek.model as MD
tok_layers = []
orig = MD.Block.forward
def hook(self, x, start_pos, pre_mix, positions=None):
    o = orig(self, x, start_pos, pre_mix, positions=positions)
    h = o[0] if isinstance(o, tuple) else o
    tok_layers.append((self.layer_id, h.detach().clone()))
    return o
MD.Block.forward = hook
e.reset()
for p in range(N):
    e._forward_chunk(ids[:, p:p+1], p)
torch.cuda.synchronize()
MD.Block.forward = orig

print("batch layers=%d tok layers=%d" % (len(batch_layers), len(tok_layers)), flush=True)
# token 路径每层只保留最后一个 token 那次
for i in range(4):
    bt = batch_layers[i][2].float()[0, N-1]
    tt = tok_layers[(N-1)*4 + i][1].float()[0, 0]
    d = float((bt - tt).abs().max())
    print("L%d: batch absmax=%8.3f  token absmax=%8.3f  maxdiff=%.6f" % (
          i, float(bt.abs().max()), float(tt.abs().max()), d), flush=True)
