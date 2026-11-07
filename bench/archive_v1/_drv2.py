import subprocess, time, json, urllib.request, os
env = dict(os.environ); env['PYTHONPATH']='src'; env['PYTHONUNBUFFERED']='1'
p = subprocess.Popen([r'.venv\Scripts\python.exe','-u',r'E:\Develop\Deepseek\deepseek\_tracer2.py'],
                     cwd=r'E:\Develop\Deepseek\tinyinfer', env=env,
                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
try:
    for _ in range(40):
        time.sleep(3)
        try:
            urllib.request.urlopen('http://127.0.0.1:18104/health', timeout=3); break
        except Exception: continue
    body = json.dumps({'model':'DeepSeek-V4.1-Flash-1017','messages':[{'role':'user','content':'The capital of France is'}],'max_tokens':3,'temperature':0}).encode()
    req = urllib.request.Request('http://127.0.0.1:18104/v1/chat/completions', data=body, headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req, timeout=300) as r: out = json.loads(r.read())
    print('RESULT content:', repr(out['choices'][0]['message']['content']))
finally:
    p.terminate()
    try: p.wait(timeout=10)
    except Exception: p.kill()
    for line in (p.stdout.read() if p.stdout else '').splitlines():
        if line.startswith('[T]'): print(line)