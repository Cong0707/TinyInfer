import sys
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer import resources as R
from tinyinfer.weights import ShardStore

for name, md, n in (("Qwen 0.8B", r"E:\Models\Qwen3.5-0.8B", 24),
                    ("Qwen 27B", r"E:\Models\Qwen3.8-27B", 64)):
    st = ShardStore(md)
    p = R.model_profile(st)
    p["topk"] = 0
    p["tied_head"] = False
    pl = R.plan(None, n, model=p)
    print(f"{name}: layer {p['layer_bytes']/2**20:.0f}MB | resident {pl.gpu_layers}/{n} | "
          f"cache {pl.cache_bytes/2**30:.1f}GB | stream {pl.stream_layers}")
    print(f"   {pl.reason[:150]}")
    print()
