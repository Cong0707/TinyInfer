
import os, sys
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
os.environ["TI_FUSED_STATS"] = "1"
import torch
from tinyinfer import engine
from tinyinfer.config import TinyConfig

# 只构造一个 MoE 层，跑一次 decode 形状的 forward，看日志里 fused=?
import inspect
print("engine file:", engine.__file__)
src = inspect.getsource(engine)
import re
for m in re.finditer(r"TI_[A-Z_]+", src):
    pass
envs = sorted(set(re.findall(r"TI_[A-Z0-9_]+", src)))
print("engine env knobs:", envs[:20])
