import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.kernels.sparse_attn import sparse_attn_cuda
import inspect
print('RESULT sig:', inspect.signature(sparse_attn_cuda))
src_ = inspect.getsource(sparse_attn_cuda)
print('RESULT first 1400 chars:')
print(src_[:1400])