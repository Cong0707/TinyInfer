RES = 1223.8 / 3981 * 1e6 / 1024
print('RESULT avg per H2D call: %.1f KB' % RES)
print('RESULT calls per decode step: %d' % (3981/3))
print('RESULT calls per layer: %d' % (3981/3/40))
# 如果每次是 6 个专家 x 每专家 6 个张量
print('RESULT 6 experts x 6 tensors = %d per layer' % 36)
# 实际数据传输量
GB_per_token = 3981/3 * (1223.8/3981) / 1000
print('RESULT bytes per token: %.2f GB' % GB_per_token)
print('RESULT at 23.8 GB/s: %.0f ms' % (GB_per_token/23.8*1000))