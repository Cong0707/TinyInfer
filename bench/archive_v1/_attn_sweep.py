
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.kernels import native
from tinyinfer.nn import sparse_attn_ref

torch.manual_seed(0)
def run(tag, B, M, H, D, N, K, use_neg1, rand_sink):
    q = torch.randn(B, M, H, D, dtype=torch.bfloat16, device="cuda")
    kv = torch.randn(B, N, D, dtype=torch.bfloat16, device="cuda")
    sink = torch.randn(H, dtype=torch.float32, device="cuda") if rand_sink else torch.zeros(H, dtype=torch.float32, device="cuda")
    idxs = torch.stack([torch.randperm(N)[:K] for _ in range(M)]).unsqueeze(0).int().cuda()
    if use_neg1: idxs[0, 0, -3:] = -1
    scale = 1.0/(D**0.5)
    ref = sparse_attn_ref(q, kv, sink, idxs, scale)
    out = torch.empty_like(q)
    fn = native.kernel("sparse_attn_sink", "sparse_attn_sink")
    blk = 128; smem = 3*4 + 32*4
    native.launch(fn, (B*M, H), (blk,), [q, kv, sink, idxs, out, B, M, H, D, N, K, float(scale)], shared=smem)
    torch.cuda.synchronize()
    d = (out.float()-ref.float()).abs().max().item()
    print(f"{tag:32s} maxdiff={d:.3e}")

run("基础 sink=0",         1,1,1,64,32,4, False, False)
run("sink 随机",           1,1,1,64,32,4, False, True)
run("带 -1 填充",          1,1,1,64,32,4, True,  False)
run("带 -1 + 随机 sink",   1,1,1,64,32,4, True,  True)
run("H=4",                 1,1,4,64,32,16, True, True)
run("M=2",                 1,2,4,64,32,16, True, True)
run("D=512 (真实)",        1,1,4,512,32,16, True, True)
run("K=64",                1,1,4,64,64,64, True, True)
