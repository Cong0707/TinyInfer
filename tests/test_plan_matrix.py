"""规划器矩阵脚本是「可移植性契约」，保持它的输出机器可读。

矩阵本身在 scripts/plan_matrix.py：3 份真实模型 x 从 GB300 到 GTX 1650 的机器谱系。
这个测试只锁输出格式和最基本的不变式 —— 具体的规划判据在
tests/test_planner_hardware.py 和 scripts/planner_compare.py（带 --check）。
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "plan_matrix.py"
NEEDED = ("model", "machine", "n_layers", "resident", "stream", "slots",
          "cache_gb", "floor_s", "fits", "gpu")


def _rows():
    # env 必须继承：Windows 上缺 SystemRoot/PATH 时子进程起不来，stdout 会是 None。
    import os

    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    out = subprocess.run([sys.executable, str(SCRIPT), "--json"],
                         capture_output=True, text=True, timeout=300, env=env)
    assert out.returncode == 0, out.stderr[-800:]
    return json.loads(out.stdout)


def test_matrix_json_is_well_formed():
    rows = _rows()
    # 多模型 x 多机器，至少十几个组合
    assert isinstance(rows, list) and len(rows) >= 12
    models = {r["model"] for r in rows}
    assert len(models) >= 3, "矩阵至少要覆盖三份模型"
    names = [r["machine"] for r in rows]
    assert any("GB300" in n for n in names), "要覆盖到数据中心卡"
    assert any("8GB" in n for n in names), "要覆盖到小卡"
    assert any("1650" in n or "2080" in n for n in names), "要覆盖到老卡"
    for r in rows:
        for key in NEEDED:
            assert key in r, (r.get("machine"), key)
        assert r["resident"] + r["stream"] == r["n_layers"], r
        assert r["slots"] >= 0, r


def test_no_combination_raises():
    """求解器在任何 (模型, 机器) 上都不该抛异常 —— 装不下也要给出降级布局。"""
    rows = _rows()
    errs = [r for r in rows if "error" in r]
    assert not errs, errs[:3]


def test_floor_seconds_is_positive():
    """每一步至少要搬点东西（除非全部常驻且槽位覆盖整步）。"""
    for r in _rows():
        assert r["floor_s"] is not None and r["floor_s"] >= 0
