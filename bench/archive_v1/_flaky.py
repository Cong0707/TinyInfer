import subprocess, sys, os, time
env = dict(os.environ)
env['PYTHONPATH'] = 'src'
env['PYTHONUNBUFFERED'] = '1'
results = []
for i in range(3):
    t0 = time.perf_counter()
    r = subprocess.run([r'.venv\Scripts\python.exe', '-u', 'scripts/check_kernels.py'],
                       cwd=r'E:\Develop\Deepseek\tinyinfer', env=env,
                       capture_output=True, text=True, timeout=900)
    dt = time.perf_counter() - t0
    last = [l for l in r.stdout.splitlines() if 'passed' in l or 'FAIL' in l]
    rc = r.returncode
    print(f'RESULT run{i+1}: rc={rc} {dt:.0f}s  {last[-1] if last else "no summary"}')
    if len(last) > 1:
        for l in last[1:4]: print(f'    {l.strip()}')
    results.append((rc, last[-1] if last else ''))
print('RESULT stable:', len(set(results)) == 1)