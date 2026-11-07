import sys
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from transformers import AutoModelForCausalLM  # noqa
import json

# What does the engine actually do per token on a machine with enough VRAM?
# Read the fp4 path: does it dequantise every token, or once at load?
from tinyinfer import moe
import inspect
src = inspect.getsource(moe)
i = src.find("def forward_fp4")
print("=== Expert.forward_fp4 ===")
print(src[i:i+1800] if i > 0 else "not found")
