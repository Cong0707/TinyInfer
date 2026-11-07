
import sys
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
import torch
import tinyinfer._moe_C_stable_libtorch  # noqa
from tinyinfer.vllm_ops import gate_topk

E, TOPK, SCALE = 384, 6, 1.5
torch.manual_seed(0)
ok_i, ok_w = True, 0.0
for T in (1, 4, 17):
    gating = torch.randn(T, E, device="cuda", dtype=torch.float32)
    bias = torch.randn(E, device="cuda", dtype=torch.float32)
    fused = gate_topk(gating, TOPK, True, SCALE, bias)
    assert fused is not None, "gate_topk returned None"
    w, idx = fused
    scores = torch.nn.functional.softplus(gating).sqrt()
    order = torch.argsort(scores + bias, dim=-1, descending=True, stable=True)
    ref_idx = order[..., :TOPK]
    ref_w = scores.gather(1, ref_idx)
    ref_w = ref_w / (ref_w.sum(-1, keepdim=True) + 1e-20) * SCALE
    same = torch.equal(idx.long(), ref_idx)
    diff = (w - ref_w).abs().max().item()
    print(f"T={T:3d}  idx_equal={same}  w_maxdiff={diff:.3e}")
    ok_i = ok_i and same
    ok_w = max(ok_w, diff)
print("ALL IDX EQUAL:", ok_i, " max w diff:", f"{ok_w:.3e}")

# benchmark
def bench(fn, warm=20, it=200):
    for _ in range(warm): fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(it): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / it * 1000.0
for T in (1, 8, 64):
    gating = torch.randn(T, E, device="cuda", dtype=torch.float32)
    bias = torch.randn(E, device="cuda", dtype=torch.float32)
    def torch_route():
        s = torch.nn.functional.softplus(gating).sqrt()
        o = torch.argsort(s + bias, dim=-1, descending=True, stable=True)
        i = o[..., :TOPK]
        v = s.gather(1, i)
        v = v / (v.sum(-1, keepdim=True) + 1e-20) * SCALE
        return v, i
    tv = bench(lambda: gate_topk(gating, TOPK, True, SCALE, bias))
    tt = bench(torch_route)
    print(f"T={T:3d}  vLLM {tv:7.2f} us   torch {tt:7.2f} us   {tt/tv:5.2f}x")
