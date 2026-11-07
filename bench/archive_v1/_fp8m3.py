import sys, torch, time
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.kernels import native
from tinyinfer.deepseek import gemv_fp4_marlin as M
torch.cuda.init(); torch.zeros(1, device='cuda')
N, K = 2304, 5120
SYM = ('_ZN6marlin6MarlinILx2814749767172868ELx562949953487106ELx1125899906909960'
       'ELx2814749767106568ELi256ELi1ELi8ELi8ELb1ELi4ELi2ELb0EEEvPK4int4S3_PS1_S4_S3_PKf'
       'S3_S6_S3_iiiiPibbbi')
p = torch.cuda.get_device_properties(0)
native.set_dynamic_smem('marlin_fp4', SYM, int(p.shared_memory_per_block_optin))
launch = native.bind('marlin_fp4', SYM, (int(p.multi_processor_count),), (256,),
                     'TTOnnnTnni'+'iii'+'Ti'+'iii', shared=int(p.shared_memory_per_block_optin))
smem = int(p.shared_memory_per_block_optin)
w = torch.randint(0,256,(N,K//2),dtype=torch.uint8,device='cuda')
s = torch.randint(120,132,(N,K//32),dtype=torch.uint8,device='cuda')
wt = M.MarlinFp4Weight(w, s)
# 参考：自研 fp8 GEMV
from tinyinfer.deepseek.fp4_native import gemv_fp4_pq, quantize_act
x = torch.randn(1,K,dtype=torch.bfloat16,device='cuda')
try:
    aq, asc = quantize_act(x, K)
    ws = M._workspace(x.device)
    out = torch.empty(1, N, dtype=torch.bfloat16, device='cuda')
    launch(aq.view(torch.uint8), wt.qweight, out, None, None, None, wt.scales,
           asc.float().reshape(1,-1), None, 1, N, K, aq.stride(0), ws, 0,0,0, smem)
    torch.cuda.synchronize()
    ref = gemv_fp4_pq(aq[0], asc[0], w, s)
    d = (out.float()-ref.float()).abs()
    print(f'RESULT fp8 marlin out {tuple(out.shape)}')
    print(f'RESULT maxdiff {d.max().item():.4f}  rel {d.max().item()/ref.abs().max().item():.2e}')
    print(f'RESULT ref[:3] {ref[0,:3].tolist()}  out[:3] {out[0,:3].tolist()}')
except Exception as e:
    import traceback; traceback.print_exc()
    print('RESULT FAILED', repr(e)[:150])