"""Are the WEIGHTS still moving while the gate is flat?

Two very different worlds produce the same flat gate series:
  (a) the update moves the policy a lot and the moves cancel out -> noise, bigger rounds help
  (b) the update barely moves the policy at all -> advantage collapse / clipping / KL anchor,
      and bigger rounds are the WRONG fix
This measures ||theta_b - theta_a|| relative to ||theta_a||, on CPU.
"""
import os, sys, math
from safetensors import safe_open

work = sys.argv[1]
pairs = [(sys.argv[i], sys.argv[i + 1]) for i in range(2, len(sys.argv), 2)]


def load(d):
    for fn in ("model.safetensors",):
        p = os.path.join(d, fn)
        if os.path.exists(p):
            return p
    raise SystemExit("no safetensors in " + d)


def dist(da, db):
    fa, fb = load(da), load(db)
    num = 0.0
    den = 0.0
    per = []
    with safe_open(fa, framework="pt") as A, safe_open(fb, framework="pt") as B:
        keys = [k for k in A.keys() if k in B.keys()]
        for k in keys:
            a = A.get_tensor(k).float()
            b = B.get_tensor(k).float()
            if a.shape != b.shape:
                continue
            d2 = (b - a).pow(2).sum().item()
            n2 = a.pow(2).sum().item()
            num += d2
            den += n2
            if n2 > 0:
                per.append((math.sqrt(d2 / n2), k))
    per.sort(reverse=True)
    return math.sqrt(num), math.sqrt(den), per


for a, b in pairs:
    da, db = os.path.join(work, a), os.path.join(work, b)
    if not (os.path.isdir(da) and os.path.isdir(db)):
        print("skip %s -> %s (missing)" % (a, b))
        continue
    d, n, per = dist(da, db)
    print("%-16s -> %-16s   ||dtheta|| = %.4f   ||theta|| = %.1f   relative = %.3e"
          % (a, b, d, n, d / n))
    for r, k in per[:3]:
        print("      top-moved  %-60s  %.3e" % (k, r))
