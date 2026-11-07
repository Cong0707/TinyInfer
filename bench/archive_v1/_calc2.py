# 2654 次传输，808ms。8GB卡上，每次传输多少？
n = 2654
ms = 808.0
bw = 25.5e9
total_gb = ms/1000 * bw / 2**30
print(f'RESULT 2654 次传输 {total_gb:.2f} GB，平均每次 {total_gb*2**30/n/2**20:.2f} MB')
print()
# 理论：
LAYER = 204*2**20
EXP = 18.8*2**20
print(f'层: 30 层 x 204MB = {30*LAYER/2**30:.2f}GB  (但每层 214MB？)')
print(f'专家: 240 x 18.8MB = {240*EXP/2**30:.2f}GB')
print(f'合计 {30*LAYER/2**30 + 240*EXP/2**30:.2f} GB')
print()
print('如果实际传了 20.6GB，说明有重复传输')
print(f'  - 层被传两次? {2*30*LAYER/2**30:.1f}GB')
print(f'  - 或专家传两次 + 层一次 = {(30*LAYER + 2*240*EXP)/2**30:.1f}GB')
print(f'  - 或层传两次 + 专家一次 = {(2*30*LAYER + 240*EXP)/2**30:.1f}GB')