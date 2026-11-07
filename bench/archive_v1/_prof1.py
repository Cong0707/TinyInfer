import sys
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.tp import Device
from tinyinfer.resources import model_profile
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
prof = model_profile(d)
print('profile:', {k: v for k, v in list(prof.items())[:12]})