"""运行时自适应：PID 式闭环调优。

离线训练（tinyinfer.tune）是扫一遍预设挑最好的，它的局限在于是**开环**的：
选完参数就不管了。但引擎运行时的状态一直在变 —— 预热进度、页缓存温度、
别的进程抢内存、路由分布漂移。同一组参数在冷启动和跑了一千步之后不是一回事。

所以需要一个闭环：**测量 -> 算误差 -> 调参数 -> 再测量**。这就是 PID。

## 控制哪两个量

被控量选「每步里 IO 占了多少时间」，因为它是引擎当前最贵的一项（本机实测
单步 2.4s 里有 1~2s 在读盘），而且可以直接测：

    io_ratio = 本步读盘耗时 / 本步总耗时

控制量选「预取提前几层」（prefetch_depth）。这个数原来是写死的 2 层：
提前太少，层的专家还没读到 GPU 就已经要算了；提前太多，预取的专家把缓存
挤掉正在用的。最优点随负载和盘速变，所以它应该被调，不该被写死。

## 为什么用 PID 而不是简单的阈值

纯阈值（"IO 超过 30% 就加一层"）会在边界上来回横跳，而且有惯性：调大之后
要过几步才能看到效果，期间阈值又触发一次，最后振荡。PID 的积分项吸收这个
延迟，微分项抑制过冲。参数不需要很准 —— 这里控制的是一个单调、低维、
有明确边界的量，P 主导、I 消除稳态偏差就够。
"""
from __future__ import annotations


class PID:
    """离散 PID，输出裁剪到 [lo, hi]。

    anti-windup：积分项在输出饱和时不再累积，否则一旦被顶到边界，回程要
    很多步才能退出来（这是 PID 最常见的实际故障）。
    """

    def __init__(self, kp: float, ki: float, kd: float,
                 lo: float = 0.0, hi: float = 1.0, target: float = 0.0):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.lo = float(lo)
        self.hi = float(hi)
        self.target = float(target)
        self.integral = 0.0
        self.prev_err = None
        self.last_out = None

    def reset(self, target: float | None = None) -> None:
        if target is not None:
            self.target = float(target)
        self.integral = 0.0
        self.prev_err = None

    def step(self, measured: float, dt: float = 1.0) -> float:
        """喂一个测量值，返回下一步的控制量。"""
        dt = max(1e-6, float(dt))
        err = self.target - float(measured)
        deriv = 0.0 if self.prev_err is None else (err - self.prev_err) / dt
        cand_integral = self.integral + err * dt
        out = self.kp * err + self.ki * cand_integral + self.kd * deriv
        # 输出饱和时不积累积分（anti-windup）
        if self.lo < out < self.hi:
            self.integral = cand_integral
        out = min(self.hi, max(self.lo, out))
        self.prev_err = err
        self.last_out = out
        return out


class PrefetchController:
    """用 IO 占比控制预取深度。

    目标：每步的 IO 时间占比压到 target_ratio 附近。太高说明预取不够（或者
    盘真的慢），调大深度；太低说明预取过度、白占缓存。

    输出是连续的（0..1），映射到整数深度时用四舍五入并加死区，避免在相邻
    两个深度之间每步横跳。
    """

    def __init__(self, target_ratio: float = 0.30, depth_lo: int = 0, depth_hi: int = 4,
                 kp: float = 6.0, ki: float = 1.5, kd: float = 0.5):
        self.depth_lo = int(depth_lo)
        self.depth_hi = int(depth_hi)
        self.pid = PID(kp, ki, kd, lo=0.0, hi=1.0, target=float(target_ratio))
        self.depth = max(self.depth_lo, min(self.depth_hi, 2))
        self.samples = 0
        self.history: list[tuple[float, int]] = []

    def observe(self, io_ms: float, step_ms: float, *, min_step_ms: float = 5.0) -> int:
        """喂一步的测量，返回更新后的预取深度。"""
        if step_ms < min_step_ms:
            return self.depth
        ratio = max(0.0, min(1.0, float(io_ms) / float(step_ms)))
        out = self.pid.step(ratio)
        # 连续输出 -> 整数深度。0.5 附近加死区，避免横跳。
        want = self.depth_lo + out * (self.depth_hi - self.depth_lo)
        new_depth = int(round(want))
        if abs(new_depth - self.depth) <= 0 and abs(want - (self.depth + 0.5)) > 0.35:
            # 离当前深度不够远，保持不动（死区）
            new_depth = self.depth
        self.depth = max(self.depth_lo, min(self.depth_hi, new_depth))
        self.samples += 1
        if len(self.history) < 256:
            self.history.append((ratio, self.depth))
        return self.depth

    def summary(self) -> dict:
        if not self.history:
            return {"samples": 0}
        ratios = [r for r, _ in self.history]
        return {
            "samples": self.samples,
            "io_ratio_mean": round(sum(ratios) / len(ratios), 3),
            "io_ratio_last": round(ratios[-1], 3),
            "depth": self.depth,
            "target": self.pid.target,
        }
