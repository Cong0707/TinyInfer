import subprocess, sys, os
for n in (2, 3, 8, 40):
    r = subprocess.run([sys.executable, "/tmp/t1.py", str(n)], capture_output=True, text=True, timeout=2400)
    out = (r.stdout + r.stderr)
    got = [l for l in out.splitlines() if l.startswith("R n=")]
    print(got[0] if got else ("n=%d FAIL %s" % (n, (out.strip().splitlines() or ["?"])[-1][:120])), flush=True)
