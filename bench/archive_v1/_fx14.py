import pathlib
p = pathlib.Path(r'E:\\Develop\\Deepseek\\deepseek\\_b90dll.bat')
t = p.read_text(encoding='utf-8', errors='ignore')
t = t.replace('-Xcompiler "/Zc:__cplusplus"', '-Xcompiler "/Zc:__cplusplus /wd2719"')
p.write_text(t, encoding='utf-8', newline=chr(13)+chr(10))
print('patched:', '/wd2719' in t)