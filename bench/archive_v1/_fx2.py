import pathlib
p = pathlib.Path(r'E:\\Develop\\Deepseek\\deepseek\\_mma_test.cu')
t = p.read_text(encoding='utf-8')
t = t.replace('#include "cutlass/arch/mma_sm89.h"' + chr(10) + '#include "cutlass/numeric_types.h"' + chr(10), '')
t = t.replace('  using Mma = cutlass::arch::Mma<cutlass::gemm::GemmShape<16, 8, 32>, 32,' + chr(10) +
              '      cutlass::float_e4m3_t, cutlass::layout::RowMajor,' + chr(10) +
              '      cutlass::float_e4m3_t, cutlass::layout::ColumnMajor,' + chr(10) +
              '      float, cutlass::layout::RowMajor, cutlass::arch::OpMultiplyAdd>;' + chr(10), '')
t = t.replace('  typename Mma::FragmentC d;', '  float d[4];')
p.write_text(t, encoding='utf-8', newline=chr(10))
print('cutlass refs left:', t.count('cutlass'))