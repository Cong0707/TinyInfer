import pathlib
p = pathlib.Path(r'E:\\Develop\\Deepseek\\tinyinfer\\src\\tinyinfer\\csrc\\fp8_gemm.cu')
t = p.read_text(encoding='utf-8')
t = t.replace('#include <cute/tensor.hpp>', '#include <cutlass/util/packed_stride.hpp>' + chr(10) + '#include <cute/tensor.hpp>')
t = t.replace('using MmaTileShape = Shape<_128, _128, _128>;',
              '// Tile K must equal the scale granularity K (a static_assert in' + chr(10) +
              '// sm120_blockwise_mma_builder.inl requires size<2>(TileShape) == ScaleGranularityK),' + chr(10) +
              '// and the checkpoints store one scale per 32 along K. So the K tile is 32.' + chr(10) +
              'using MmaTileShape = Shape<_128, _128, _32>;')
t = t.replace('  if (M <= 0 || N <= 0 || K <= 0 || (K % 32) != 0 || (N % 32) != 0) return -2;',
              '  if (M <= 0 || N <= 0 || K <= 0 || (K % 32) != 0 || (N % 32) != 0) return -2;')
p.write_text(t, encoding='utf-8', newline=chr(10))
print('patched', t.count('packed_stride'), t.count('_32>;'))