
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
