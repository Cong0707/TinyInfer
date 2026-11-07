import os, subprocess, sys, time
env = dict(os.environ)
env['PYTHONPATH'] = 'src'
for frac in ('0.28', '0.20', '0.15', '0.10'):
    e = dict(env); e['TI_SLAB_FRAC'] = frac; e['PYTHONUNBUFFERED'] = '1'
    r = subprocess.run([r'.venv\Scripts\python.exe', '-u', r'E:\Develop\Deepseek\deepseek\_speed.py'],
                       cwd=r'E:\Develop\Deepseek\tinyinfer', env=e,
                       capture_output=True, text=True, timeout=300)
    line = [l for l in r.stdout.splitlines() if 'FINAL' in l]
    print(f'RESULT frac={frac}: {line[-1] if line else "no output"}')
    sys.stdout.flush()