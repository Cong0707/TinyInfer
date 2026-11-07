import subprocess, time, json, urllib.request, os, sys
env = dict(os.environ)
env['PYTHONPATH'] = 'src'
env['PYTHONUNBUFFERED'] = '1'
p = subprocess.Popen([r'.venv\Scripts\python.exe', '-u', '-m', 'tinyinfer.cli', 'serve',
                      '--backend', 'tiny', '--model-dir', r'E:\models\DeepSeek-V4.1-Flash-1017',
                      '--port', '18100'],
                     cwd=r'E:\Develop\Deepseek\tinyinfer', env=env,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    for _ in range(40):
        time.sleep(3)
        try:
            urllib.request.urlopen('http://127.0.0.1:18100/health', timeout=3)
            break
        except Exception:
            continue
    body = json.dumps({'model': 'DeepSeek-V4.1-Flash-1017',
                       'messages': [{'role':'user','content':'The capital of France is'}],
                       'max_tokens': 5, 'temperature': 0}).encode()
    req = urllib.request.Request('http://127.0.0.1:18100/v1/chat/completions',
                                 data=body, headers={'Content-Type': 'application/json'})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as r:
        out = json.loads(r.read())
    dt = time.perf_counter() - t0
    txt = out['choices'][0]['message']['content']
    print(f'RESULT chat ok in {dt:.0f}s: {txt!r}')
    print(f'RESULT usage: {out.get("usage")}')
finally:
    p.terminate()
    try: p.wait(timeout=10)
    except Exception: p.kill()