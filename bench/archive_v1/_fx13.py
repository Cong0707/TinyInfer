import pathlib
p = pathlib.Path(r'E:\\Develop\\Deepseek\\deepseek\\_fp8_gemm_sm90.cu')
t = p.read_text(encoding='utf-8')
# 用 3.9 的正确 API，先做非 blockwise 的最小版验证工具链
minimal = t.replace('cutlass::detail::Sm90BlockwiseScaleConfig<ScaleGranularityM, ScaleGranularityN, ScaleGranularityK>',
                    'cutlass::detail::Sm90BlockwiseScaleConfig<ScaleGranularityM, ScaleGranularityN, ScaleGranularityK>')
# 替换 stride 构造
t = t.replace(
    '  stride_A = typename Gemm::GemmKernel::StrideA{}(cute::make_stride(K, 1, M * K));' + chr(10) +
    '  stride_B = typename Gemm::GemmKernel::StrideB{}(cute::make_stride(1, K, N * K));' + chr(10) +
    '  stride_C = typename Gemm::GemmKernel::StrideC{}(cute::make_stride(N, 1, M * N));' + chr(10) +
    '  stride_D = typename Gemm::GemmKernel::StrideD{}(cute::make_stride(N, 1, M * N));',
    '  stride_A = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideA{}, {M, K, 1});' + chr(10) +
    '  stride_B = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideB{}, {N, K, 1});' + chr(10) +
    '  stride_C = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideC{}, {M, N, 1});' + chr(10) +
    '  stride_D = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideD{}, {M, N, 1});')
p.write_text(t, encoding='utf-8', newline=chr(10))
print('api count', t.count('make_cute_packed_stride'))