import subprocess, time, json, urllib.request, os
env = dict(os.environ); env['PYTHONPATH']='src'; env['PYTHONUNBUFFERED']='1'
p = subprocess.Popen([r'.venv\Scripts\python.exe','-u','-m','tinyinfer.cli','serve',
                      '--backend','tiny','--model-dir',r'E:\models\DeepSeek-V4.1-Flash-1017',
                      '--port','18105'], cwd=r'E:\Develop\Deepseek\tinyinfer', env=env,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    for _ in range(40):
        time.sleep(3)
        try:
            urllib.request.urlopen('http://127.0.0.1:18105/health', timeout=3); break
        except Exception: continue
    # 同样的内容，一个走 chat 一个走 completions，但 prompt 用完全一样的文本
    chat_body = json.dumps({'model':'DeepSeek-V4.1-Flash-1017',
        'messages':[{'role':'user','content':'Answer in one word. The capital of France is'}],
        'max_tokens':3,'temperature':0}).encode()
    comp_body = json.dumps({'model':'DeepSeek-V4.1-Flash-1017',
        'prompt':'<|begin_of_sentence|><|User|>Answer in one word. The capital of France is<|Assistant|>',
        'max_tokens':3,'temperature':0}).encode()
    for path, body, kind in (('/v1/chat/completions', chat_body, 'chat'),
                             ('/v1/completions', comp_body, 'completions')):
        try:
            req = urllib.request.Request('http://127.0.0.1:18105'+path, data=body, headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(req, timeout=300) as r: out = json.loads(r.read())
            txt = out['choices'][0]['message']['content'] if kind=='chat' else out['choices'][0]['text']
            print(f'RESULT {kind}: {txt!r}')
        except Exception as e:
            print(f'RESULT {kind} FAILED: {repr(e)[:80]}')
finally:
    p.terminate()
    try: p.wait(timeout=10)
    except Exception: p.kill()