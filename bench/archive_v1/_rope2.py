import inspect
from transformers.models.qwen3_5 import modeling_qwen3_5 as M
src = inspect.getsource(M.Qwen3_5TextRotaryEmbedding)
i = src.find("def __init__")
print(src[i:i+1400])
