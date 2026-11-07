import sys, torch, os
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
os.environ['TI_MTP'] = '1'
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
p = tok('The capital of France is', return_tensors='pt').input_ids[0].tolist()
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(10)))
# 直接测 _draft
with torch.inference_mode():
    eng.reset()
    logits = eng.forward(torch.tensor([p], device='cuda'), 0)
    nid = int(logits[0,-1].argmax())
    print('RESULT nid:', nid, 'pos:', len(p))
    drafts = eng._draft(nid, len(p), 5)
    print('RESULT drafts:', drafts)
    # verify 一下
    vl = eng.forward(torch.tensor([drafts], device='cuda'), start_pos=len(p))
    preds = [logits[0,-1]] + [vl[0,t] for t in range(len(drafts)-1)]
    matches = [int(preds[t].argmax()) == drafts[t] for t in range(len(drafts))]
    print('RESULT per-position match:', matches)