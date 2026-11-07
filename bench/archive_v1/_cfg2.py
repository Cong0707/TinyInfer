import json, pathlib, sys
d = pathlib.Path(r'E:\\models\\DeepSeek-V4.1-Flash-1017')
cfgj = json.load(open(d/'config.json'))
tc = cfgj.get('text_config', cfgj)
for k in ('routed_scaling_factor','norm_topk_prob','n_activated_experts','scoring_func','num_experts_per_tok'):
    print(f'RESULT {k} = {tc.get(k)}')