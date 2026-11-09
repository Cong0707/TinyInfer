"""启动期自动调优：引擎在新机器上第一次运行就找到自己的最优配置。"""

import argparse, json, os, subprocess, sys, time
from pathlib import Path

NL = chr(10)
_MAX_CANDIDATES = 3
_PROBE_TIMEOUT = 150


def _src():
    return str(Path(__file__).resolve().parents[1])


def _key(model_dir):
    try:
        from .tune import machine_key
        mk = machine_key()
    except Exception:
        mk = "unknown"
    return mk + "|model=" + Path(str(model_dir)).name


def _code(model_dir, env_extra):
    L = []
    a = L.append
    a("import sys, os, json, time, statistics")
    a("sys.path.insert(0, %r)" % _src())
    a("for _k, _v in %r.items(): os.environ[_k] = _v" % ({k: str(v) for k, v in env_extra.items()},))
    a("import torch")
    a("from tinyinfer.engine import Engine")
    a("out = {}")
    a("e = None")
    a("try:")
    a("    e = Engine.from_hf(%r, device='cuda', allow_missing_layers=True)" % str(model_dir))
    a("    ids = torch.tensor([[100,200,300,400,500]], dtype=torch.long, device='cuda')")
    a("    s = []")
    a("    ok = True")
    a("    with torch.inference_mode():")
    a("        e.forward(ids, 0); torch.cuda.synchronize()")
    a("        for _ in range(3):")
    a("            e.forward(torch.tensor([[7]], device='cuda'), 6); torch.cuda.synchronize()")
    a("        for _ in range(3):")
    a("            t0 = time.perf_counter()")
    a("            e.forward(ids, 0); torch.cuda.synchronize()")
    a("            s.append((time.perf_counter()-t0)*1e3)")
    a("        x = e.forward(ids, 0); torch.cuda.synchronize()")
    a("        y = e.forward(ids, 0); torch.cuda.synchronize()")
    a("    if not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(y).all()):")
    a("        ok = False")
    a("    else:")
    a("        d = float((x.float()-y.float()).abs().max())")
    a("        if d > 1e-2 * (float(x.float().abs().max()) + 1e-6): ok = False")
    a("    s.sort()")
    a("    out['step_ms'] = statistics.median(s) if s else 0.0")
    a("    out['ok'] = ok")
    a("except Exception as exc:")
    a("    out['error'] = repr(exc)[:300]")
    a("try:")
    a("    out['slots'] = int(e.expert_store.slots.slots) if e is not None else -1")
    a("except Exception:")
    a("    out['slots'] = -1")
    a("try:")
    a("    if e is not None: e.close()")
    a("except Exception:")
    a("    pass")
    a("print('AT' + json.dumps(out))")
    return NL.join(L)


def _probe(model_dir, env_extra, timeout_s):
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", _src())
    env.update({k: str(v) for k, v in env_extra.items()})
    try:
        p = subprocess.run([sys.executable, "-c", _code(model_dir, env_extra)],
                           capture_output=True, text=True, timeout=timeout_s, env=env)
    except subprocess.TimeoutExpired:
        return {"error": "timeout"}
    for line in (p.stdout or "").splitlines():
        if line.startswith("AT"):
            try:
                return json.loads(line[2:])
            except Exception:
                pass
    tail = (p.stderr or "").strip().splitlines()
    return {"error": (tail[-1] if tail else "no output")[:200]}


def _candidates():
    hi = 24
    try:
        from .resources import plan
        hi = max(1, int(plan(model={}).expert_slots))
    except Exception:
        pass
    lo = max(1, min(4, hi))
    if hi <= lo:
        return [hi]
    n = min(_MAX_CANDIDATES, hi - lo + 1)
    step = (hi - lo) / (n - 1)
    return sorted({int(round(lo + i * step)) for i in range(n)})


def discover(model_dir, *, budget_s=120.0, constraints=None, force=False, verbose=True):
    constraints = dict(constraints or {})
    model_dir = str(model_dir)
    key = _key(model_dir)
    if not force:
        try:
            from .tune import load_tuned
            c = (((load_tuned().get("entries") or {}).get(key) or {}).get("chosen") or {}).get("autotuned")
            if isinstance(c, dict) and c.get("params"):
                return c
        except Exception:
            pass
    t0 = time.perf_counter()
    cands = [int(constraints["expert_slots"])] if "expert_slots" in constraints else _candidates()
    if verbose:
        print("[autotune] candidates=%s" % (cands,), flush=True)
    measured, best = [], None
    for s in cands:
        left = budget_s - (time.perf_counter() - t0)
        if left < 20:
            break
        r = _probe(model_dir, {"TI_SLOTS": str(s)}, min(_PROBE_TIMEOUT, left))
        r["candidate_slots"] = s
        measured.append(r)
        if verbose:
            msg = r.get("error") or ("step=%.0fms ok=%s" % (r.get("step_ms", 0), r.get("ok")))
            print("[autotune] slots=%-3d %s" % (s, msg), flush=True)
        if r.get("ok") and r.get("step_ms"):
            if best is None or r["step_ms"] < best["step_ms"]:
                best = r
    el = time.perf_counter() - t0
    if best is None:
        return {"params": {}, "measured": measured, "error": "none passed determinism", "elapsed_s": round(el, 1)}
    res = {"params": {"expert_slots": int(best["candidate_slots"])}, "step_ms": round(float(best["step_ms"]), 1),
           "measured": measured, "elapsed_s": round(el, 1), "constraints": constraints,
           "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if verbose:
        print("[autotune] best slots=%d step=%.0fms (%.0fs)" % (best["candidate_slots"], best["step_ms"], el), flush=True)
    try:
        from .tune import load_tuned, save_tuned
        d = load_tuned()
        e = d.setdefault("entries", {}).setdefault(key, {})
        e.setdefault("chosen", {})["autotuned"] = res
        e["ts"] = res["ts"]
        save_tuned(d)
    except Exception:
        pass
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--budget", type=float, default=120.0)
    ap.add_argument("--constraint", action="append", default=[])
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    cons = {}
    for kv in a.constraint:
        if "=" in kv:
            k, v = kv.split("=", 1)
            try:
                cons[k] = int(v)
            except ValueError:
                cons[k] = v
    print(json.dumps(discover(a.model, budget_s=a.budget, constraints=cons, force=a.force), ensure_ascii=False)[:1500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
