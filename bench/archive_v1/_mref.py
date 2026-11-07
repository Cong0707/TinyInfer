import sys
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
import torch, inspect
import tinyinfer.torch_compat
from transformers.models.qwen3_5 import modeling_qwen3_5 as M
src = inspect.getsource(M.Qwen3_5TextModel.forward)
i = src.find("create_causal_mask")
print(src[max(0,i-300):i+900])
