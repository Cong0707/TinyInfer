import sys, torch
sys.path.insert(0, "/models/tinyinfer/src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
M = "/models/deepseek"
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
pad = tok.encode("你好", add_special_tokens=False)[0]
ids = torch.tensor([[pad]], device="cuda")
e = Engine(M, tier=OptLevel.AGGRESSIVE, device="cuda", layers=[0,1], tokenizer=tok)

# 在捕获期把 indices 和 sids 拷进持久 buffer（copy_ 合法）
OBS = {}
import tinyinfer.deepseek.moe as MO
orig = MO.MoE._forward_gpu_only
def spy(self, x, xf, indices, weights):
    lid = self.layer_id
    cap = torch.cuda.is_current_stream_capturing()
    if not cap and lid not in OBS:
        OBS[lid] = {"idx": torch.zeros(64, dtype=torch.int64, device="cuda"),
                    "sid": torch.zeros(64, dtype=torch.int32, device="cuda")}
    r = orig(self, x, xf, indices, weights)
    o = OBS.get(lid)
    if o is not None:
        n = int(indices.numel())
        try:
            o["idx"][:n].copy_(indices.view(-1)[:n])
            sid = getattr(self, "_sids_buf", None)
            if sid is not None:
                o["sid"][:n].copy_(sid[:n])
        except Exception:
            pass
    return r
MO.MoE._forward_gpu_only = spy

with torch.inference_mode():
    e.forward(ids, 0); e.forward(ids, 0); torch.cuda.synchronize()
    e.pin_hot_experts(); torch.cuda.synchronize()
    e.reset(); e.forward(ids, 1); torch.cuda.synchronize()
    print("GRAPH %.4g" % float(e._graph["out"].abs().max()), flush=True)
    for lid, o in sorted(OBS.items()):
        print("GB L%d idx=%s sid=%s" % (lid, o["idx"][:6].tolist(), o["sid"][:6].tolist()), flush=True)
    sl = e.expert_store.slots
    tab = sl._slot_tab.cpu()
    print("TAB-cpu L0=%s" % ([(i, int(tab[0][i])) for i in range(384) if int(tab[0][i]) >= 0],), flush=True)
    print("TAB-cpu L1=%s" % ([(i, int(tab[1][i])) for i in range(384) if int(tab[1][i]) >= 0],), flush=True)
