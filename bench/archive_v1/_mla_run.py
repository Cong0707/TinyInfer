
import torch
import tinyinfer._C_stable_libtorch  # noqa

op = getattr(torch.ops._C, "sm100_cutlass_mla_decode", None)
print("op present:", op is not None)
ws = getattr(torch.ops._C, "sm100_cutlass_mla_get_workspace_size", None)
print("ws op present:", ws is not None)

# DeepSeek MLA 形状: q_nope [B, H, 512], q_pe [B, H, 64], cache [pages, 128, 576]
dev = "cuda"
B, H, D_lat, D_rope = 1, 128, 512, 64
P, PS = 16, 128
dtype = torch.bfloat16
try:
    out = torch.empty(B, H, D_lat, dtype=dtype, device=dev)
    lse = torch.empty(B, H, dtype=torch.float32, device=dev)
    q_nope = torch.randn(B, H, D_lat, dtype=dtype, device=dev)
    q_pe = torch.randn(B, H, D_rope, dtype=dtype, device=dev)
    kv = torch.randn(P, PS, D_lat + D_rope, dtype=dtype, device=dev)
    seq_lens = torch.tensor([P * PS], dtype=torch.int32, device=dev)
    page_table = torch.arange(P, dtype=torch.int32, device=dev).view(1, P)
    size = int(ws(P * PS, B, 26, 1))
    workspace = torch.empty(size, dtype=torch.uint8, device=dev)
    torch.ops._C.sm100_cutlass_mla_decode(
        out, lse, q_nope, q_pe, kv, seq_lens, page_table, workspace, 0.135, 1)
    torch.cuda.synchronize()
    print("MLA RUN OK  out[0,0,:3]:", [round(v, 3) for v in out[0, 0, :3].tolist()])
    # 对比参考：q_nope @ cache[:N, :512].T 的 softmax 加权（只验证量级，不是精确）
    print("  finite:", bool(torch.isfinite(out).all()), " lse[0,0]:", float(lse[0, 0]))
except Exception as exc:
    print("MLA FAILED:", type(exc).__name__, str(exc)[:500])
