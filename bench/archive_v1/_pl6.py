import sys
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.resources import model_profile, plan, Hardware, Card
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
prof = model_profile(d)
hw = Hardware(name='x', vram_gb=24, ram_gb=64, ram_avail_gb=56, cards=[Card(0,'x',24,23)])
p = plan(hw=hw, n_layers=40, model=prof, speculation=True)
print('RESULT Plan fields:', [f for f in dir(p) if not f.startswith('_')])