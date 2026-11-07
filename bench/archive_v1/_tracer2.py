import sys, os
sys.path.insert(0, r'E:\Develop\Deepseek\tinyinfer\src')
import torch
import tinyinfer.server as S
_orig = S.TinyModelBackend._generate_locked
def pg(self, request):
    prompt_text = self.render_prompt(request)
    ids = self._encode(prompt_text)
    print(f'[T] ids={ids}', flush=True)
    # 手工走一遍 forward，看第一个 token
    with torch.inference_mode():
        lg = self._model.forward(torch.tensor([ids], device='cuda'), 0)
        first = int(lg[0,-1].argmax())
        print(f'[T] manual first={first} ({self._decode([first])!r})', flush=True)
    for ev in _orig(self, request):
        print(f'[T] emitted tok={getattr(ev,"token_id",None)} text={getattr(ev,"text",None)!r}', flush=True)
        yield ev
S.TinyModelBackend._generate_locked = pg
from tinyinfer.cli import main
sys.argv = ['tinyinfer','serve','--backend','tiny','--model-dir',r'E:\models\DeepSeek-V4.1-Flash-1017','--port','18104']
main()