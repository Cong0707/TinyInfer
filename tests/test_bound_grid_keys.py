"""`BoundLaunch` 的 grid 是固化的：缓存键必须覆盖 grid 里的每个运行时变量。

这一类 bug 在本工程出现过三次：

1. `_MOE_FUSED["act"]` —— grid 含 `n_exp = rows*topk`，按名字缓存
2. `_MOE_FUSED["down"]` —— grid 含 `_rows`，按名字缓存
3. （历史）`kv_norm_rope_quant`、`bt_gemv` 等后来改成了按几何键缓存

后果：11 行 prefill 之后，所有 1 行 decode 都用错的 grid，kernel 越界
读写同一个大 buffer 内的合法地址 —— 数值全错，但 compute-sanitizer 报 0 错误。

这条测试用源码扫描锁住规则：**bound launcher 的缓存键必须包含 grid 里出现的
每个变量名**。做不到静态验证全部情况，但能抓住「用字面量字符串当键」这种。
"""

import re

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "src" / "tinyinfer"


def _read(rel):
    p = ROOT / rel
    return p.read_text(encoding="utf-8") if p.exists() else ""


def test_no_bound_launch_cached_under_a_bare_string_with_variable_grid():
    """找 `X.get("name")` + `BoundLaunch(..., grid含变量, ...)` + `X["name"] = ` 三件套。"""
    offenders = []
    for rel in ("deepseek/moe.py", "deepseek/hc.py", "deepseek/attention.py",
                "deepseek/compressed_attention.py", "gemm.py", "kernels/sparse_attn.py"):
        src = _read(rel)
        if not src:
            continue
        # 每处 BoundLaunch(...) 的 grid 参数（第 2 个位置参数）
        for m in re.finditer(r"BoundLaunch\s*\(([^)]*)\)", src, re.S):
            args = m.group(1)
            # 粗略取第二段（grid）
            parts = [p.strip() for p in args.split(",")]
            grid = parts[1] if len(parts) > 1 else ""
            if not re.search(r"[a-zA-Z_][a-zA-Z0-9_]*", grid):
                continue
            # grid 里出现的标识符（排除纯数字和 self.cfg 之类）。
            # 先把字符串字面量剔掉，否则 kernel 名会被当成变量。
            _g = re.sub("\u0022[^\u0022]*\u0022|\u0027[^\u0027]*\u0027", "", grid)
            names = set(re.findall(r"\b([a-z_][a-z0-9_]{1,20})\b", _g))
            names -= {"self", "cfg", "int", "dim", "and", "or", "kernel"}
            if not names:
                continue
            # 找这附近的缓存键：.get("literal") 或 [("literal")]
            window = src[max(0, m.start() - 700):m.start()]
            keys = re.findall(r"\.get\(\"([a-z_0-9]+)\"\)|\[\"([a-z_0-9]+)\"\]", window)
            flat = [k for t in keys for k in t if k]
            if flat and not any(n in flat for n in names):
                offenders.append((rel, flat[-1], sorted(names)))
    assert not offenders, (
        "bound launcher cached under a key that omits its grid variables: %s" % offenders)
