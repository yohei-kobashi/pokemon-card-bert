import os
p = "/root/mrl_convert.py"
s = open(p).read(); orig = s

s = s.replace('ap.add_argument("--temp", type=float, default=0.5)',
              'ap.add_argument("--temp", type=float, default=0.5)\n'
              'ap.add_argument("--qmin", type=float, default=0.0,\n'
              '                help="drop pairs whose playout advantage |qw-ql| is below this. "\n'
              '                     "The Q estimate from 24 playouts has an SE around 0.2 and the "\n'
              '                     "median pair margin is 0.26, so most pairs are coin flips -- and "\n'
              '                     "measured, they do not merely add nothing: training on all of them "\n'
              '                     "moved held-out conformance 54.3 -> 53.6, while >=0.35 moved it "\n'
              '                     "52.1 -> 58.1. The low-margin majority outvotes the signal.")')
assert "qmin" in s

old = """        qw, ql = float(d["qw"]), float(d["ql"])
        rw, rl = float(d.get("rww") or 0.0), float(d.get("rwl") or 0.0)
        if abs(qw - ql) < 1e-9 and abs(rw - rl) < 1e-9:
            n_flat += 1
            continue"""
new = """        qw, ql = float(d["qw"]), float(d["ql"])
        rw, rl = float(d.get("rww") or 0.0), float(d.get("rwl") or 0.0)
        if abs(qw - ql) < 1e-9 and abs(rw - rl) < 1e-9:
            n_flat += 1
            continue
        if abs(qw - ql) < a.qmin:
            n_weak += 1
            continue"""
assert s.count(old) == 1
s = s.replace(old, new)
s = s.replace("n_in = n_out = n_same = n_flat = 0",
              "n_in = n_out = n_same = n_flat = n_weak = 0")
s = s.replace('print("[mrl] %d pairs -> %d rows (same-text %d, flat %d) | beta %.2f temp %.2f"\n'
              '      % (n_in, n_out, n_same, n_flat, a.beta, a.temp))',
              'print("[mrl] %d pairs -> %d rows (same-text %d, flat %d, below-qmin %d) '
              '| beta %.2f temp %.2f qmin %.2f"\n'
              '      % (n_in, n_out, n_same, n_flat, n_weak, a.beta, a.temp, a.qmin))')
assert s != orig and "n_weak" in s
open(p + ".tmp", "w").write(s); os.replace(p + ".tmp", p)
print("patched mrl_convert.py with --qmin")
