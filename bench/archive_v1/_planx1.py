import sys
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.resources import plan, model_profile, Hardware, Card, detect
from tinyinfer.loading import ShardIndex
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
p = model_profile(ShardIndex(d))
real = detect()
cases = [
    ('本机 RTX 5060 8GB', real),
    ('H200 141GB x1', Hardware('H200', 141.0, 1024.0, 1000.0, cards=[Card(0,'H200',141.0,141.0)])),
    ('H200 141GB x4', Hardware('H200x4', 564.0, 2048.0, 2000.0, cards=[Card(i,'H200',141.0,141.0) for i in range(4)])),
]
for name, hw in cases:
    pl = plan(hw, p['n_layers'], model=p)
    print(f'{name}:')
    print('   ', pl.reason[:190])
    print('    slots=', pl.card_slots, ' layer_card set=', len(set(pl.layer_card.values())))