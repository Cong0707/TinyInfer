import sys, torch, time
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.kernels import native
# 找 act_quant 的 launch 方式
import tinyinfer.deepseek.fp4_native as FN
import inspect
print('RESULT quantize_act:', str(inspect.signature(FN.quantize_act))[:120])
print(inspect.getsource(FN.quantize_act)[:800])