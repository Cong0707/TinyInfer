import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.weights import ShardStore
from tinyinfer.gemm import SCALE_FMT, SCALE_DTYPE, _act_quant, _bf16_default
from tinyinfer.kernels import native
import kernel as _k
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
st = ShardStore(d)
name = 'layers.3.attn.wq_b'
w = st.get(name + '.weight').to('cuda'); s = st.get(name + '.scale').to('cuda')
N, K = w.shape
# 用 fp64 做绝对参考
def ref_fp64(Aq, AS, w, ws):
    Af = Aq.view(torch.float8_e4m3fn).double()
    Wf = w.view(torch.float8_e4m3fn).double()
    asf = torch.pow(2.0, (AS.view(torch.uint8).to(torch.int32) - 127).to(torch.float64))
    wsf = torch.pow(2.0, (ws.view(torch.uint8).to(torch.int32) - 127).to(torch.float64))
    out = torch.zeros(N, dtype=torch.float64, device='cuda')
    nbk = K//32
    for kb in range(nbk):
        p = (Af[kb*32:(kb+1)*32] * Wf[:, kb*32:(kb+1)*32]).sum(dim=1)
        out += p * asf[kb] * wsf[(torch.arange(N, device='cuda')//32), kb]
    return out
torch.manual_seed(0)
x = torch.randn(1, K, dtype=torch.bfloat16, device='cuda')
aq, asc = _act_quant(x, 32, SCALE_FMT, SCALE_DTYPE)
# native simt
y_simt = torch.empty(N, device='cuda', dtype=torch.bfloat16)
native.launch(native.kernel('fp8_gemv_bs','fp8_gemv_bs'), ((N+15)//16,), (512,),
              [aq.view(torch.uint8).reshape(-1), asc.view(torch.uint8).reshape(-1),
               w.view(torch.uint8).reshape(-1), s.view(torch.uint8).reshape(-1), y_simt, N, K])
torch.cuda.synchronize()
ref = ref_fp64(aq.reshape(-1), asc.reshape(-1), w, s)
e_simt = (y_simt.double() - ref).abs().max().item()
print(f'RESULT simt vs fp64: maxabs={e_simt:.4e}  (bf16 ulp at that scale ~ {ref.abs().max().item()*2**-8:.2e})')