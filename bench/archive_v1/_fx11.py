import pathlib
p = pathlib.Path(r'E:\\Develop\\Deepseek\\deepseek\\_fp8_gemm_sm90.cu')
t = p.read_text(encoding='utf-8')
t = t.replace('#include <cutlass/util/packed_stride.hpp>' + chr(10), '')
old = '''  typename Gemm::GemmKernel::StrideA stride_A;'''
# 用 make_stride 直接构造 packed stride（row-major A[M,K] => (K,1), col-major B[N,K] => (1,K)）
t = t.replace(
    '  stride_A = cutlass::make_cute_packed_stride(stride_A, cute::make_shape(M, K, 1));' + chr(10) +
    '  stride_B = cutlass::make_cute_packed_stride(stride_B, cute::make_shape(N, K, 1));' + chr(10) +
    '  stride_C = cutlass::make_cute_packed_stride(stride_C, cute::make_shape(M, N, 1));' + chr(10) +
    '  stride_D = cutlass::make_cute_packed_stride(stride_D, cute::make_shape(M, N, 1));',
    '  stride_A = typename Gemm::GemmKernel::StrideA{}(cute::make_stride(K, 1, M * K));' + chr(10) +
    '  stride_B = typename Gemm::GemmKernel::StrideB{}(cute::make_stride(1, K, N * K));' + chr(10) +
    '  stride_C = typename Gemm::GemmKernel::StrideC{}(cute::make_stride(N, 1, M * N));' + chr(10) +
    '  stride_D = typename Gemm::GemmKernel::StrideD{}(cute::make_stride(N, 1, M * N));')
p.write_text(t, encoding='utf-8', newline=chr(10))
print('patched')