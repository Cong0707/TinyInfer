import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.kernels import native
from tinyinfer.official_kernel_ref import fp4_act_quant as ref
torch.manual_seed(0)
def mine(x, block, scale_dtype, inplace):
    from tinyinfer.kernels.fp4q import fp4_act_quant
    return fp4_act_quant(x, block, inplace, scale_dtype)
for (shape, block, sd, inplace) in (
    ((1,512), 32, torch.float8_e8m0fnu, False),
    ((2,3,512), 32, torch.float8_e8m0fnu, False),
    ((1,256), 16, torch.float8_e4m3fn, False),
    ((1,512), 32, torch.float8_e8m0fnu, True),
    ((1,256), 16, torch.float8_e4m3fn, True),
):
    x = torch.randn(*shape, dtype=torch.bfloat16, device='cuda')
    x[0].view(-1)[:block*2] = 0.0
    a = x.clone(); b = x.clone()
    ra = ref(a, block, inplace, sd)
    rb = mine(b, block, sd, inplace)
    if inplace:
        d = (ra.view(torch.uint8) != rb.view(torch.uint8)).sum().item()
        print(f'RESULT blk{block} inplace bf16 diff={d}/{ra.numel()}')
    else:
        rq, rs = ra; mq, ms = rb
        dq = (rq.view(torch.uint8) != mq.view(torch.uint8)).sum().item()
        ds = (rs.view(torch.uint8) != ms.view(torch.uint8)).sum().item()
        print(f'RESULT blk{block} fp4 diff={dq}/{rq.numel()} scale diff={ds}/{rs.numel()}')