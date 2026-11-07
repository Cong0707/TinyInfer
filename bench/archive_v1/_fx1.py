import pathlib
p = pathlib.Path(r'E:\\Develop\\Deepseek\\deepseek\\_probe_cutlass.cu')
t = p.read_text(encoding='utf-8')
t = t.replace('GemmShape<128, 128, 128>', 'GemmShape<128, 64, 128>')
t = t.replace('GemmShape<64, 64, 128>', 'GemmShape<64, 32, 128>')
p.write_text(t, encoding='utf-8', newline=chr(10))
print('ok')