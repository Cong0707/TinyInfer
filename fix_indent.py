import io
p = r"E:\Develop\Deepseek\tinyinfer\src\tinyinfer\loading.py"
lines = open(p, encoding="utf-8").read().splitlines()
# 定位 3716 行（1-based）到 3738 行之间的坏块
start = None
for i, l in enumerate(lines):
    if "_cap = getattr(self.ram, \"capacity_entries\", 0) or 0" in l:
        start = i
        break
print("start line", start + 1)
# 找 end: 包含 "if _admit_ok:" 的行之后一行
end = None
for i in range(start, min(start + 60, len(lines))):
    if lines[i].strip() == "if _admit_ok:":
        end = i
        break
print("end line", end + 1)
new = [
    '                    _cap = getattr(self.ram, "capacity_entries", 0) or 0',
    '                    # prefill 过去是 cache=False：故意不填 RAM 缓存，怕把 decode 热集挤掉。',
    '                    # 但这条在大内存机器上是纯损失：L20 有 371GB 内存、缓存容量',
    '                    # 13150 项，专家总共 15360 项 —— 装得下绝大部分。prefill 每层读',
    '                    # 240 个专家、40 层约 9600 次读盘全被丢弃，随后 decode 再读一遍。',
    '                    # 实测：prefill 单次 126.5 秒读了 14896 次盘（约 280GB）。',
    '                    # 判据：缓存容量装得下模型 75% 以上时，prefill 也 admit。',
    '                    _cache_all = bool(cache)',
    '                    if not _cache_all and self.ram is not None:',
    '                        try:',
    '                            _cap_e = int(getattr(self.ram, "capacity_entries", 0) or 0)',
    '                            _tot_e = int(getattr(self.ram, "n_layers", 0) or 0) * int(',
    '                                getattr(self.ram, "n_experts", 0) or 0)',
    '                            if _tot_e <= 0:',
    '                                _tot_e = 40 * 384',
    '                            if _cap_e >= int(_tot_e * 0.75):',
    '                                _cache_all = True',
    '                        except Exception:',
    '                            pass',
    '                    _admit_ok = _cache_all and (not _cap or len(experts) <= _cap)',
]
out = lines[:start] + new + lines[end:]
open(p, "w", encoding="utf-8").write("\n".join(out) + "\n")
import ast
ast.parse(open(p, encoding="utf-8").read())
print("AST_OK")