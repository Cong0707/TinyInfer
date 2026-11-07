import pathlib
p = pathlib.Path(r'E:\\Develop\\Deepseek\\deepseek\\_build_dll.bat')
t = p.read_text(encoding='utf-8', errors='ignore')
t = t.replace('%INC%', '-DCUTLASS_DEBUG_TRACE_LEVEL=2 %INC%')
p.write_text(t, encoding='utf-8', newline=chr(13)+chr(10))
print('trace on')