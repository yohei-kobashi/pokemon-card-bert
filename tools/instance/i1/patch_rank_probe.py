"""rank_probe: report the loss against its FLOOR and its UNIFORM baseline, and add --limit.

The raw number the probe printed (1.5945 -> 1.5770 over two epochs) is a soft cross-entropy that
CONTAINS the target's own entropy, so "it barely moved" was an unreadable statement. Measured on
the 59,974 usable branch points at tau=0.5:

    uniform predictor   E[log K]      = 1.5773      (K = 5.09 candidates on average)
    entropy floor       E[H(target)]  = 1.4273
    reducible gap                     = 0.1500

so 1.5945 is WORSE than uniform and 1.5770 is exactly AT uniform: two epochs closed ~0% of the
gap. Printing the excess over the floor makes that legible instead of hiding it in the third
decimal of a number whose floor nobody checked.

Also adds a within-epoch loss trace -- an epoch AVERAGE cannot show whether the fit is improving
inside the epoch -- and --limit, so a learning-rate sweep costs minutes rather than hours.
"""
import os
import re

P = os.path.join(os.getcwd(), "tools/rank_probe.py")
s = open(P).read()

if "--limit" in s:
    print("already patched")
    raise SystemExit(0)

# 1) CLI
A = '    ap.add_argument("--seed", type=int, default=0)'
B = A + '''
    ap.add_argument("--limit", type=int, default=0,
                    help="cap usable branch points (0 = all) -- for fast lr sweeps")
    ap.add_argument("--trace-every", type=int, default=500,
                    help="print a running train loss every N branch points")'''
assert s.count(A) == 1, "seed arg"
s = s.replace(A, B)

# 2) limit + benchmarks right after loading
A2 = '''    rng = random.Random(args.seed)
    rng.shuffle(data)'''
B2 = '''    rng = random.Random(args.seed)
    rng.shuffle(data)
    if args.limit:
        data = data[:args.limit]
        print(f"limited to {len(data)} branch points", flush=True)
    # The loss is a soft cross-entropy, so its FLOOR is the target's own entropy and a
    # uniform predictor already scores E[log K]. Without both, the raw number is unreadable.
    import math as _m
    _H, _U = [], []
    for _p, _c, _i, _q in data:
        _e = [_m.exp(x / args.tau) for x in _q]
        _Z = sum(_e)
        _pr = [x / _Z for x in _e]
        _H.append(-sum(x * _m.log(x + 1e-12) for x in _pr))
        _U.append(_m.log(len(_q)))
    FLOOR, UNIF = sum(_H) / len(_H), sum(_U) / len(_U)
    print(f"loss benchmarks: entropy FLOOR {FLOOR:.4f} | UNIFORM {UNIF:.4f} | "
          f"reducible gap {UNIF - FLOOR:.4f}", flush=True)'''
assert s.count(A2) == 1, "rng anchor"
s = s.replace(A2, B2)

# 3) within-epoch trace
A3 = '''            tot += float(loss); nb += 1'''
B3 = '''            tot += float(loss); nb += 1
            if args.trace_every and nb % args.trace_every == 0:
                _w = min(nb, args.trace_every)
                _recent = (tot - _prev_tot) / _w
                print(f"    [{nb:6d}] recent loss {_recent:.4f} "
                      f"(uniform {UNIF:.4f}, floor {FLOOR:.4f}, "
                      f"gap closed {100.0 * (UNIF - _recent) / max(1e-9, UNIF - FLOOR):+.1f}%)",
                      flush=True)
                _prev_tot = tot'''
assert s.count(A3) == 1, "accumulator anchor"
s = s.replace(A3, B3)

A4 = '''        tot, nb = 0.0, 0'''
B4 = '''        tot, nb = 0.0, 0
        _prev_tot = 0.0'''
assert s.count(A4) == 1, "tot init"
s = s.replace(A4, B4)

# 4) epoch line reports the excess too
A5 = '''        print(f"epoch {ep}: train loss {tot/max(1,nb):.4f} | held-out "
              f"E[Q(top) - mean Q(others)] = {m:+.4f} +/- {se:.4f} (t={m/se:+.1f})", flush=True)'''
B5 = '''        _tl = tot / max(1, nb)
        print(f"epoch {ep}: train loss {_tl:.4f} "
              f"[floor {FLOOR:.4f}, uniform {UNIF:.4f}, gap closed "
              f"{100.0 * (UNIF - _tl) / max(1e-9, UNIF - FLOOR):+.1f}%] | held-out "
              f"E[Q(top) - mean Q(others)] = {m:+.4f} +/- {se:.4f} (t={m/se:+.1f})", flush=True)'''
assert s.count(A5) == 1, "epoch print"
s = s.replace(A5, B5)

open(P, "w").write(s)
print("patched", P)
