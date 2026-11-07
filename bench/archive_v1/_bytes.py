
import sys
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.loading import ShardIndex
from tinyinfer.resources import model_profile
from tinyinfer.adapters import open_adapter
a = open_adapter(r"E:\models\DeepSeek-V4.1-Flash-1017")
p = model_profile(ShardIndex(a.store.model_dir))
GB = 2**30
print("n_layers      :", p.get("n_layers"))
print("layer_bytes   :", round(p["layer_bytes"]/GB,3), "GiB  (每层非专家)")
print("max_layer     :", round(p["max_layer_bytes"]/GB,3), "GiB")
print("expert_bytes  :", round(p["expert_bytes"]/GB,4), "GiB (每个专家)")
print("embed         :", round(p["embed_bytes"]/GB,3))
print("head          :", round(p["head_bytes"]/GB,3))
c = a.load_config()
print("n_routed_experts:", c.n_routed_experts, " topk:", c.n_activated_experts, " layers:", c.n_layers)
L = c.n_layers; E = c.n_routed_experts; K = c.n_activated_experts
dense_tok = L * p["layer_bytes"] + p["embed_bytes"] + p["head_bytes"]
exp_tok   = L * K * p["expert_bytes"]
print(f"\n每 token 读量:")
print(f"  dense/层部分 : {dense_tok/GB:.2f} GiB")
print(f"  专家({L}x{K}={L*K} 个) : {exp_tok/GB:.2f} GiB")
print(f"  合计         : {(dense_tok+exp_tok)/GB:.2f} GiB")
