import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
eng = Engine(r"E:\Develop\Deepseek\tinyinfer\cache\weights", max_layers=4, device="cuda", gpu_layers=2)
ids = [19923, 14, 342, 1030]
with torch.inference_mode():
    eng.forward(torch.tensor([ids], device="cuda"))
    eng.forward(torch.tensor([[22651]], device="cuda"), start_pos=4)
print("streamed 2 resident + 2 streamed ok; devices:",
      [eng._layer_device(eng.block(i)) for i in range(4)])
