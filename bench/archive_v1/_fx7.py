import pathlib
p = pathlib.Path(r'E:\\Develop\\Deepseek\\tinyinfer\\src\\tinyinfer\\kernels\\cutlass_gemm.py')
t = p.read_text(encoding='utf-8')
t = t.replace("here = pathlib.Path(__file__).resolve().parents[2]", "here = pathlib.Path(__file__).resolve().parents[3]  # repo root: src/tinyinfer/kernels/x.py")
p.write_text(t, encoding='utf-8', newline=chr(10))
print('patched')