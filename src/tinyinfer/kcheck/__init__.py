"""Kernel conformance harness.

Every native kernel in this repo is supposed to be bit-identical to a reference
implementation; several were only checked by hand once and that is how bugs like the
signed-zero fp4 encoding and the e8m0 exponent offset survived. This module gives each
kernel a place to state its claim and a way to fail loudly when it stops being true.

A test registers a case with `check(...)`, which compares a produced tensor against an
expected one and records the result. `run_all()` reports every failure at once rather
than stopping at the first, because a quantiser regression usually affects several
shapes together.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class Case:
    name: str
    detail: str
    ok: bool


@dataclass
class Report:
    cases: list = field(default_factory=list)

    def add(self, case: Case) -> None:
        self.cases.append(case)

    @property
    def failed(self):
        return [c for c in self.cases if not c.ok]

    def summary(self) -> str:
        bad = self.failed
        head = f'{len(self.cases) - len(bad)}/{len(self.cases)} kernel cases passed'
        if not bad:
            return head
        lines = [head, '']
        for c in bad:
            lines.append(f'  FAIL {c.name}: {c.detail}')
        return chr(10).join(lines)


def _as_bytes(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.uint8).reshape(-1)


def check_bitwise(report: Report, name: str, got: torch.Tensor,
                  want: torch.Tensor, note: str = '') -> bool:
    """Exact match. Use this for quantisers and encoders, where "close" is not the claim."""
    if got.shape != want.shape:
        report.add(Case(name, f'shape {tuple(got.shape)} != {tuple(want.shape)}', False))
        return False
    g, w = _as_bytes(got), _as_bytes(want)
    bad = int((g != w).sum())
    if bad:
        i = int((g != w).nonzero()[0])
        extra = f' ({note})' if note else ''
        report.add(Case(name, f'{bad}/{g.numel()} elements differ, first at {i}{extra}',
                        False))
        return False
    report.add(Case(name, note, True))
    return True


def check_close(report: Report, name: str, got: torch.Tensor,
                want: torch.Tensor, rtol: float = 2e-2, note: str = '') -> bool:
    """Relative match, for kernels whose only claim is "within fp32/bf16 rounding"."""
    g = got.float().reshape(-1)
    w = want.float().reshape(-1)
    if g.numel() != w.numel():
        report.add(Case(name, f'{g.numel()} vs {w.numel()} elements', False))
        return False
    scale = w.abs().max().clamp_min(1e-12)
    worst = ((g - w).abs() / scale).max().item()
    ok = worst <= rtol
    report.add(Case(name, f'max relative error {worst:.3e} <= {rtol:.1e} {note}', ok))
    return ok


def run_all() -> int:
    """Entry point for scripts/check_kernels.py; returns a process exit code."""
    from . import (attention, cache, cases, chat, elementwise, hc, moe_gemv, more, mtp,
                   planner, slots)

    report = Report()
    attention.register(report)
    cache.register(report)
    cases.register(report)
    elementwise.register(report)
    more.register(report)
    moe_gemv.register(report)
    chat.register(report)
    slots.register(report)
    hc.register(report)
    planner.register(report)
    mtp.register(report)
    print(report.summary())
    return 0 if not report.failed else 1
