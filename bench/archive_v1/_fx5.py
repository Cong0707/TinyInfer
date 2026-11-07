import pathlib
p = pathlib.Path(r'E:\\Develop\\Deepseek\\deepseek\\_p2.cu')
t = p.read_text(encoding='utf-8')
t = t.replace('using LayoutB = cutlass::layout::RowMajor;', 'using LayoutB = cutlass::layout::ColumnMajor;  // B is stored [N,K]; ColumnMajor makes it [K,N] to cutlass')
p.write_text(t, encoding='utf-8', newline=chr(10))
print('ok')