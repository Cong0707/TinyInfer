import sys, torch, time
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.kernels import native
torch.cuda.init(); torch.zeros(1, device='cuda')
N, K = 2304, 5120
SYM = ('_ZN6marlin6MarlinILx2814749767172868ELx562949953487106ELx1125899906909960'
       'ELx2814749767106568ELi256ELi1ELi8ELi8ELb1ELi4ELi2ELb0EEEvPK4int4S3_PS1_S4_S3_PKf'
       'S3_S6_S3_iiiiPibbbi')
props = torch.cuda.get_device_properties(0)
sms = int(props.multi_processor_count)
smem = int(props.shared_memory_per_block_optin)
try:
    native.set_dynamic_smem('marlin_fp8', SYM, smem)
    b = native.bind('marlin_fp8', SYM, (sms,), (256,), 'TTOnnnTnni'+'iii'+'Ti'+'iii', shared=smem)
    print('RESULT fp8 launcher bound ok')
except Exception as e:
    print('RESULT bind FAILED:', repr(e)[:180])