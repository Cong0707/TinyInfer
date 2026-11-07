import pathlib, re
root = pathlib.Path(r'E:\\Develop\\Deepseek\\tinyinfer\\tests')
targets = {
    'test_compressed_attention.py': ['test_prefill_and_decode'],
    'test_streaming_session.py': ['test_repeat_prefill_same_argmax', 'test_multi_token_matches_stepwise'],
}
for fname, tests in targets.items():
    p = root / fname
    t = p.read_text(encoding='utf-8')
    if 'requires_cuda' in t:
        print(f'{fname}: already marked'); continue
    t = t.replace('from conftest import model_dir', 'from conftest import model_dir, requires_cuda')
    if 'requires_cuda' not in t:
        # 没有 import 那行，自己插
        t = t.replace('import pytest', 'import pytest\nfrom conftest import requires_cuda', 1)
    for name in tests:
        t = re.sub(rf'^def {name}\b', f'@requires_cuda\ndef {name}', t, flags=re.M)
    p.write_text(t, encoding='utf-8', newline=chr(10))
    print(f'{fname}: marked {len(tests)}')