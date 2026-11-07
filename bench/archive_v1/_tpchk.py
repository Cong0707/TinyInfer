import sys
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer import tp
import inspect

print("=== tp.available() ===")
print(inspect.getsource(tp.available))
print("=== all_reduce_sum ===")
print(inspect.getsource(tp.all_reduce_sum))
