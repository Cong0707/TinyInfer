import ast, pathlib, sys
src = pathlib.Path(r'E:\Develop\Deepseek\tinyinfer\src\tinyinfer\server.py').read_text(encoding='utf-8')
tree = ast.parse(src)
found = 0
for node in ast.walk(tree):
    if not isinstance(node, ast.FunctionDef) or node.name != 'warmup':
        continue
    calls = [n for n in ast.walk(node) if isinstance(n, ast.Call)]
    if any('empty_cache' in ast.unparse(c) for c in calls):
        found += 1
        print('RESULT VIOLATION FOUND in warmup')
print(f'RESULT violations: {found} (0 = clean)')