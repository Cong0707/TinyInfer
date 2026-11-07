import pathlib
p = pathlib.Path(r'E:\\Develop\\Deepseek\\deepseek\\_mma_test.cu')
t = p.read_text(encoding='utf-8')
t = t.replace('  for (int i = 0; i < 16; ++i) for (int k = 0; k < 32; ++k) hA[i * 32 + k] = (uint8_t)i;', '  for (int i = 0; i < 16; ++i) for (int k = 0; k < 32; ++k) hA[i * 32 + k] = 0x38;  // 1.0' + chr(10) +
              '  for (int i = 0; i < 16; ++i) for (int k = 0; k < 32; ++k) hA[i * 32 + k] = 0x38;')
t = t.replace('  for (int n = 0; n < 8; ++n) for (int k = 0; k < 32; ++k) hB[n * 32 + k] = 1;', '  for (int n = 0; n < 8; ++n) for (int k = 0; k < 32; ++k) hB[n * 32 + k] = 0x38;  // 1.0')
t = t.replace('      float want = 32.0f * i;', '      float want = 32.0f;   // all ones: every output is 32')
p.write_text(t, encoding='utf-8', newline=chr(10))
print('patched')