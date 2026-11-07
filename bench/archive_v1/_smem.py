
import torch
p = torch.cuda.get_device_properties(0)
print("name:", p.name)
print("capability:", p.major, p.minor)
print("shared_per_block:", p.shared_memory_per_block)
print("shared_per_block_optin:", getattr(p, "shared_memory_per_block_optin", "n/a"))
print("shared_per_sm:", p.shared_memory_per_multiprocessor)
print("regs_per_sm:", p.regs_per_multiprocessor)
print("max_threads_per_sm:", p.max_threads_per_multi_processor)
