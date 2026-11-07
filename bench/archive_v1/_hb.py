
import sys
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.budget import budget
GB = 1 << 30
hb = budget()
print("total  GB:", hb.total / GB)
print("limit  GB:", hb.limit / GB)
print("room   GB:", hb.room() / GB)
print("stats:", hb.stats())
