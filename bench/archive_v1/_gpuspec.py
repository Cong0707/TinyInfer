
import torch
p = torch.cuda.get_device_properties(0)
print("name      :", p.name)
print("SM count  :", p.multi_processor_count)
print("mem GB    :", round(p.total_memory/2**30, 2))
print("clock MHz :", p.clock_rate/1000 if hasattr(p,'clock_rate') else '?')
print("mem clock :", getattr(p, 'memory_clock_rate', '?'))
print("L2 bytes  :", p.L2_cache_size)
