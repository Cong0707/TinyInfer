import sys
class Block:
    def find_spec(self, name, path=None, target=None):
        if name == 'tilelang' or name.startswith('tilelang.'):
            raise ImportError('blocked')
        return None
sys.meta_path.insert(0, Block())
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
import torch
from tinyinfer.kernels import native
from tinyinfer.kernels.quant import _native_ok
from tinyinfer.deepseek import fp4_native
from tinyinfer.deepseek.gemv_fp4_marlin import available as marlin_ok
from tinyinfer.gemm import _fp8_native
print('act_quant native:', _native_ok())
print('fp4_native:', fp4_native.available())
print('marlin:', marlin_ok())
print('fp8_native:', _fp8_native())
for name in ('fp4_quant','gemv_fp4_w4a8','sparse_attn_sink','act_quant','rms_norm','moe_gemv6'):
    try:
        native.kernel(name, name if name!='moe_gemv6' else 'moe_gemv6')
        print(f'  {name}: OK')
    except Exception as e:
        print(f'  {name}: FAIL {e}')