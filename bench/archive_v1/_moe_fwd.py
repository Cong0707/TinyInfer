import sys, inspect
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer import moe
src = inspect.getsource(moe)
i = src.find("def forward")
print("=== MoE forward 开头 ===")
print(src[i:i+2200])
