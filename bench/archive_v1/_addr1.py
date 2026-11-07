import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.budget import budget
b = budget()
buf = getattr(b, '_claim_buf', None)
print('RESULT claim_buf type:', type(buf))
if buf is not None:
    print('RESULT len:', len(buf) if hasattr(buf,'__len__') else 'n/a')
    print('RESULT buf id/addr:', id(buf))
    # 用 memoryview 的方式取真实地址
    try:
        mv = memoryview(buf)
        print('RESULT mv:', mv.nbytes)
    except Exception as e:
        print('RESULT mv fail:', e)
    # frombuffer 的 data_ptr
    t = torch.frombuffer(buf, dtype=torch.uint8)
    print('RESULT frombuffer data_ptr:', int(t.data_ptr()))
    print('RESULT is_pinned:', t.is_pinned())