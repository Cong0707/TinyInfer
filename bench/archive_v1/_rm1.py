import pathlib, re
p = pathlib.Path(r'E:\\Develop\\Deepseek\\tinyinfer\\src\\tinyinfer\\gemm.py')
t = p.read_text(encoding='utf-8')
start = t.find('            # record_stream: the kernel is still reading aq/asc/w_scale')
if start < 0:
    print('block not found')
else:
    end = t.find('            return y.view(1, n)', start)
    if end < 0:
        print('return not found')
    else:
        t = t[:start] + t[end:]
        p.write_text(t, encoding='utf-8', newline=chr(10))
        print('removed', end - start, 'chars')