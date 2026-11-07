# 综合最佳配置复测 3 次
import os, subprocess, sys, re
env = dict(os.environ); env['PYTHONPATH'] = 'src'; env['PYTHONUNBUFFERED'] = '1'
res = []
for i in range(3):
    r = subprocess.run([r'.venv\Scripts\python.exe', '-u', r'E:\Develop\Deepseek\deepseek\_speed.py'],
                       cwd=r'E:\Develop\Deepseek\tinyinfer', env=env,
                       capture_output=True, text=True, timeout=300)
    fin = [l for l in r.stdout.splitlines() if 'FINAL' in l]
    if fin: res.append(fin[-1])
    print(f'RESULT run{i+1}: {fin[-1] if fin else "none"}')
    sys.stdout.flush()
if res:
    ms = [float(re.search(r'FINAL (\d+) ms', x).group(1)) for x in res]
    print(f'RESULT median={sorted(ms)[len(ms)//2]}ms  best={min(ms)}ms  worst={max(ms)}ms')