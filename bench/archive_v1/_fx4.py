import pathlib
p = pathlib.Path(r'E:\\Develop\\Deepseek\\deepseek\\_mma_test.cu')
t = p.read_text(encoding='utf-8')
t = t.replace('  for (int i = 0; i < 16; ++i) for (int k = 0; k < 32; ++k) hA[i * 32 + k] = 0x38;  // 1.0' + chr(10) + '  for (int i = 0; i < 16; ++i) for (int k = 0; k < 32; ++k) hA[i * 32 + k] = 0x38;',
              '  for (int i = 0; i < 16; ++i) for (int k = 0; k < 32; ++k) hA[i * 32 + k] = (i < 8) ? 0x38 : 0x00;  // rows 0-7 = 1.0, 8-15 = 0')
t = t.replace('      float want = 32.0f;   // all ones: every output is 32', '      float want = (i < 8) ? 32.0f : 0.0f;')
p.write_text(t, encoding='utf-8', newline=chr(10))
print('patched')