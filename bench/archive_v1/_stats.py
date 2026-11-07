from pathlib import Path
root = Path(r"E:\Develop\Deepseek\repositories\vllm-csrc\csrc")
def rel(f):
    try:
        return str(f.relative_to(root))
    except ValueError:
        return str(f)
cu = [f for f in root.rglob("*.cu") if "rocm" not in rel(f)]
cuh = [f for f in root.rglob("*.cuh")]
h = [f for f in root.rglob("*.h")]
cpp = [f for f in root.rglob("*.cpp")]
total = sum(f.stat().st_size for f in cu)
print(f"vLLM CUDA: {len(cu)} .cu + {len(cuh)} .cuh + {len(h)} .h + {len(cpp)} .cpp")
print(f".cu total: {total/1024/1024:.1f} MB")
lines = sum(len(f.read_text(encoding='utf-8', errors='replace').splitlines()) for f in cu)
print(f".cu lines: {lines:,}")
print()
ours = Path(r"E:\Develop\Deepseek\tinyinfer\src\tinyinfer\native")
ocu = list(ours.glob("*.cu"))
olines = sum(len(f.read_text(encoding='utf-8', errors='replace').splitlines()) for f in ocu)
print(f"tinyinfer: {len(ocu)} .cu, {olines:,} lines")
print(f"vLLM has {lines/max(1,olines):.0f}x our CUDA line count")
