import sys
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer import resources as R
print('RESULT SLOT_FRACTION', getattr(R,'SLOT_FRACTION','?'))
print('RESULT MIN_SLOTS', getattr(R,'MIN_SLOTS','?'))
GB = 1024**2
layer_b = 195
max_layer = 730
V = 6.9*1024
used = 11*layer_b + 3*max_layer
spare = V - used
sf = getattr(R,'SLOT_FRACTION',0.9)
slots = int(spare * sf // 18.8)
need = 240
print(f'RESULT V={V/1024:.1f}GB used={used/1024:.1f}GB spare={spare/1024:.1f}GB slots={slots} need*0.15={need*0.15:.0f}')