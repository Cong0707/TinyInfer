import subprocess, time, json, urllib.request, os
env = dict(os.environ); env['PYTHONPATH']='src'; env['PYTHONUNBUFFERED']='1'
p = subprocess.Popen([r'.venv\Scripts\python.exe','-u','-m','tinyinfer.cli','serve',
                      '--backend','tiny','--model-dir',r'E:\models\DeepSeek-V4.1-Flash-1017',
                      '--port','18102'], cwd=r'E:\Develop\Deepseek\tinyinfer', env=env,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    for _ in range(40):
        time.sleep(3)
        try:
            urllib.request.urlopen('http://127.0.0.1:18102/health', timeout=3); break
        except Exception: continue
    for path, body in (
        ('/v1/chat/completions', {'model':'DeepSeek-V4.1-Flash-1017','messages':[{'role':'user','content':'The capital of France is'}],'max_tokens':5,'temperature':0}),
        ('/v1/completions', {'model':'DeepSeek-V4.1-Flash-1017','prompt':'The capital of France is','max_tokens':5,'temperature':0}),
    ):
        try:
            req = urllib.request.Request('http://127.0.0.1:18102'+path, data=json.dumps(body).encode(),
                                         headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(req, timeout=300) as r: out = json.loads(r.read())
            if path.startswith('/v1/chat'):
                txt = out['choices'][0]['message']['content']
            else:
                txt = out['choices'][0]['text']
            print(f'RESULT {path}: {txt!r}')
        except Exception as e:
            print(f'RESULT {path} FAILED: {repr(e)[:90]}')
finally:
    p.terminate()
    try: p.wait(timeout=10)
    except Exception: p.kill()
