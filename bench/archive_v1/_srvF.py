import subprocess, time, json, urllib.request, os, sys
env = dict(os.environ); env['PYTHONPATH']='src'; env['PYTHONUNBUFFERED']='1'
# 用 monkeypatch 在 server 进程里打印
wrapper = r'''
import sys
sys.path.insert(0, r'E:\Develop\Deepseek\tinyinfer\src')
import tinyinfer.server as S
_orig = S.TinyModelBackend.render_prompt
def patched(self, request):
    out = _orig(self, request)
    ids = self._encode(out)
    print(f'[TRACE] prompt={out!r}', flush=True)
    print(f'[TRACE] ids={ids}', flush=True)
    return out
S.TinyModelBackend.render_prompt = patched
_g = S.TinyModelBackend._generate_locked
def pg(self, request):
    for ev in _g(self, request):
        print(f'[TRACE] tok={ev.token_id} text={ev.text!r}', flush=True)
        yield ev
S.TinyModelBackend._generate_locked = pg
from tinyinfer.cli import main
sys.argv = ['tinyinfer','serve','--backend','tiny','--model-dir',r'E:\models\DeepSeek-V4.1-Flash-1017','--port','18103']
main()
'''
open(r'E:\Develop\Deepseek\deepseek\_tracer.py','w',encoding='utf-8').write(wrapper)
p = subprocess.Popen([r'.venv\Scripts\python.exe','-u',r'E:\Develop\Deepseek\deepseek\_tracer.py'],
                     cwd=r'E:\Develop\Deepseek\tinyinfer', env=env,
                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
try:
    for _ in range(40):
        time.sleep(3)
        try:
            urllib.request.urlopen('http://127.0.0.1:18103/health', timeout=3); break
        except Exception: continue
    body = json.dumps({'model':'DeepSeek-V4.1-Flash-1017','messages':[{'role':'user','content':'The capital of France is'}],'max_tokens':3,'temperature':0}).encode()
    req = urllib.request.Request('http://127.0.0.1:18103/v1/chat/completions', data=body, headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req, timeout=300) as r: out = json.loads(r.read())
    print('RESULT content:', repr(out['choices'][0]['message']['content']))
finally:
    p.terminate()
    try: p.wait(timeout=10)
    except Exception: p.kill()
    txt = p.stdout.read() if p.stdout else ''
    for line in txt.splitlines():
        if '[TRACE]' in line: print(line)