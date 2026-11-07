import sys
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.loading import ExpertStore
from tinyinfer.weights import ShardStore
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
store = ShardStore(d)
es = ExpertStore.from_store(store, cache_bytes=2<<30, device='cpu')
with es.experts(3, [7]) as recs:
    r = recs[0]
    print('RESULT type:', type(r).__name__)
    print('RESULT attrs:', [a for a in dir(r) if not a.startswith('_')])
    for a in ('layer','expert','key','ids','blob'):
        print(f'  {a}: {getattr(r, a, "<none>")!r}'[:90])