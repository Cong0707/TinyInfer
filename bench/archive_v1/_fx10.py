import pathlib
src = pathlib.Path(r'E:\\Develop\\Deepseek\\tinyinfer\\src\\tinyinfer\\csrc\\fp8_gemm.cu')
t = src.read_text(encoding='utf-8')
# 生成一个 sm90 版本：ArchTag Sm90 + tile 128x128x128（Hopper 的 blockwise 要求 K tile = 128）
t90 = t.replace('cutlass::arch::Sm120', 'cutlass::arch::Sm90')
t90 = t90.replace('using MmaTileShape = Shape<_128, _64, _32>;', 'using MmaTileShape = Shape<_128, _128, _128>;')
t90 = t90.replace('constexpr int ScaleGranularityM = 1;', 'constexpr int ScaleGranularityM = 128;')
t90 = t90.replace('constexpr int ScaleGranularityN = 32;', 'constexpr int ScaleGranularityN = 128;')
t90 = t90.replace('constexpr int ScaleGranularityK = 32;', 'constexpr int ScaleGranularityK = 128;')
t90 = t90.replace('Sm120BlockwiseScaleConfig', 'Sm90BlockwiseScaleConfig')
pathlib.Path(r'E:\\Develop\\Deepseek\\deepseek\\_fp8_gemm_sm90.cu').write_text(t90, encoding='utf-8', newline=chr(10))
print('written')