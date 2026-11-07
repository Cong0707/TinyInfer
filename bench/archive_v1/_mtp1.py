import sys
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.config import TinyConfig
import json, pathlib
d = pathlib.Path(r'E:\\models\\DeepSeek-V4.1-Flash-1017')
cfgj = json.load(open(d/'config.json'))
tc = cfgj.get('text_config', cfgj)
for k in ('num_nextn_predict_layers','dspark_block_size','mtp','num_hidden_layers'):
    print(f'RESULT {k} = {tc.get(k)}')
# 找 mtp shard 有多大
files = sorted(d.glob('*.safetensors'))
tot = sum(f.stat().st_size for f in files)
print(f'RESULT total model files {len(files)} shards, {tot/2**30:.1f} GB')
for f in files[-4:]:
    print(f'  {f.name} {f.stat().st_size/2**20:.0f} MB')