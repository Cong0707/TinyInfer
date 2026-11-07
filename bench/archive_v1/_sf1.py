import sys
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer import resources as R
print('RESULT SLOT_FRACTION', R.SLOT_FRACTION)
print('RESULT MIN_SLOTS', R.MIN_SLOTS)
GB = 1024**3
# 8GB 卡：V=8, 11 层 x 195MB = 2145MB, buffer 3x max_layer
layer_b = 195*2**20
max_layer = 730*2**20
V = 6.9*GB
used = 11*layer_b + 3*max_layer
spare = V - used
slots = int(spare * R.SLOT_FRACTION // (18.8*2**20))
print(f'RESULT 8GB: V={V/GB:.1f} used={used/GB:.1f} spare={spare/GB:.1f}GB slots={slots}')
need = 240
print(f'RESULT need_slots={need} 15%={need*0.15:.0f} -> slots {slots} >= {need*0.15:.0f} ? {slots >= need*0.15}')