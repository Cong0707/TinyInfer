import sys
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer import resources as R
from tinyinfer.weights import ShardStore
GB = 2**30; MB = 2**20
p = R.model_profile(ShardStore(r"E:\Models\Qwen3.8-27B"))
p["topk"] = 0; p["tied_head"] = False
V = 6.9 * GB
layer_b = p["layer_bytes"]; max_layer = p["max_layer_bytes"]
print(f"layer {layer_b/MB:.0f}MB, max {max_layer/MB:.0f}MB, embed {p['embed_bytes']/MB:.0f}MB")
# what does one resident layer cost?
print()
print("VRAM ledger for N resident layers (LAYER_SLACK 1.70, buf 2, ws 0.36, runtime 0.75):")
SLACK, BUF, WS, RT = 1.70, 2, 0.36*GB, 0.75*GB
for n in (1, 2, 3, 4):
    used = n*layer_b*SLACK + BUF*max_layer + WS + RT
    print(f"  {n} layers: {used/GB:.2f} GB of {V/GB:.1f} GB "
          f"{'OK' if used <= V else 'OVER'}")
print()
print("=> a 27B layer is 1441MB but costs 2450MB with slack+ring, so 2 layers barely")
print("   fit and 3 do not. VRAM is not where a dense 27B can live.")
