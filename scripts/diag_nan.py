"""统计 forward 的 NaN 率，并支持按配置分组对比。

用法：

    python scripts/diag_nan.py --runs 10
    python scripts/diag_nan.py --runs 10 --env TI_FLAT=1
    python scripts/diag_nan.py --runs 10 --env TI_FLAT=1 --env TI_RING=4

## 为什么需要它

这个仓库里有一个还没解决的 forward 不确定性（见 docs/未解决-不确定性.md）。
它的表现是「约 30~60% 的运行产生 NaN」，所以**单次结果什么都说明不了**。

排查过程中被小样本骗过两次：

- 把「预热进度」当成性能曲线（前几步慢，其实是后台 warm 线程还没跑完）
- 把「连续 4 次干净」当成找到根因（如果真实 NaN 率是 50%，连续 4 次干净
  的概率是 6.25%，完全可能发生）

所以判断任何改动是否有效，必须跑足够多次并算出比率。这个脚本就是干这个的。

## 输出

    runs=10  nan=4  rate=0.40  finite=6

对比两组时，看 rate 的差是否落在二项分布的噪声之外 —— 10 次样本的标准误约
是 sqrt(p(1-p)/n)，p=0.4 时约 0.15。所以两个配置的 rate 差要超过 0.3 才算
有信号。**更稳妥的做法是每组跑 20~30 次。**
"""

import argparse
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--env", action="append", default=[],
                    help="额外环境变量，形如 TI_FLAT=1，可重复")
    ap.add_argument("--model", default=os.environ.get("TI_MODEL",
                                                    r"E:\Models\DeepSeek-V4.1-Flash-1017"))
    args = ap.parse_args()

    for kv in args.env:
        if "=" not in kv:
            print("bad --env %r (want KEY=VALUE)" % kv, file=sys.stderr)
            return 2
        k, v = kv.split("=", 1)
        os.environ[k] = v

    src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
    sys.path.insert(0, src)

    import torch
    from tinyinfer.engine import Engine

    ids = torch.tensor([[100, 200, 300, 400, 500]], dtype=torch.long, device="cuda")
    nan = 0
    finite = 0
    for i in range(args.runs):
        eng = Engine.from_hf(args.model, device="cuda", allow_missing_layers=True)
        try:
            with torch.inference_mode():
                o = eng.forward(ids, 0)
                torch.cuda.synchronize()
            if bool(torch.isfinite(o).all()):
                finite += 1
            else:
                nan += 1
        except Exception as exc:
            # 崩溃也是失败的一种形态，和 NaN 分开计数会更准确，但先归到一起
            print("run %d failed: %r" % (i, exc), file=sys.stderr)
            nan += 1
        finally:
            try:
                eng.close()
            except Exception:
                pass

    total = max(1, args.runs)
    print("runs=%d  nan=%d  rate=%.2f  finite=%d" % (total, nan, nan / total, finite))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
