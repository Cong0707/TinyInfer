import pathlib
p = pathlib.Path(r'E:\\Develop\\Deepseek\\tinyinfer\\src\\tinyinfer\\kernels\\cutlass_gemm.py')
t = p.read_text(encoding='utf-8')
t = t.replace("                fn.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 3 + [ctypes.c_void_p]",
              "                fn.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 3 + [ctypes.c_void_p]" + chr(10) +
              "                err = dll.tinyinfer_fp8_last_error" + chr(10) +
              "                err.restype = ctypes.c_char_p" + chr(10) +
              "                err.argtypes = []")
t = t.replace("    if rc != 0:" + chr(10) + "        raise RuntimeError(f'tinyinfer_fp8_blockwise_gemm returned {rc}')",
              "    if rc != 0:" + chr(10) +
              "        detail = dll.tinyinfer_fp8_last_error()" + chr(10) +
              "        raise RuntimeError('tinyinfer_fp8_blockwise_gemm returned '" + chr(10) +
              "                           f'{rc}: {(detail or b"").decode("utf-8", "replace")}')")
p.write_text(t, encoding='utf-8', newline=chr(10))
print('ok')