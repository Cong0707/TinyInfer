import sys, traceback
class Block:
    def find_spec(self, name, path=None, target=None):
        if name == 'tilelang' or name.startswith('tilelang.'):
            raise ImportError('blocked')
        return None
sys.meta_path.insert(0, Block())
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
try:
    import tinyinfer.engine
    print('IMPORT ENGINE OK')
except Exception:
    print('=== FAILED ===')
    tb = traceback.format_exc().strip().splitlines()
    for line in tb[-24:]:
        print(line)