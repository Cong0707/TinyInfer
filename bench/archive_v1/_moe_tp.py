import sys
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
import inspect
from tinyinfer import tp, moe, attention

print("=== moe.py 里 all_reduce 的上下文 ===")
src = inspect.getsource(moe)
i = src.find("all_reduce_sum")
print(src[max(0,i-900):i+300])
