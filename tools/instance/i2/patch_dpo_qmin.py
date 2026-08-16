import os
p = "/root/ptcg/repo/tools/instance/dpo_teacher.py"
s = open(p).read(); orig = s

s = s.replace('    ap.add_argument("--limit", type=int, default=0)',
'''    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--qmin", type=float, default=0.0,
                    help="drop pairs whose playout advantage |qw-ql| is below this. Measured on "
                         "the CROSS-ENCODER side (instance1, rounds 7-9, 4,519 pairs): training "
                         "on every pair moved held-out conformance 54.3 -> 53.6, i.e. DOWN, "
                         "while >=0.35 moved it 52.1 -> 58.1. With 24 playouts the Q estimate's "
                         "SE is about 0.2 and the median pair margin is 0.26, so most pairs are "
                         "coin flips that outvote the informative minority. This flag exists to "
                         "test whether the same holds for the 4B, which had no filter at all.")''')
assert "--qmin" in s

old = """            if limit and len(rows) >= limit:
                break"""
new = """            if qmin and abs(rows[-1]["q_gap"]) < qmin:
                rows.pop()
                drop["below_qmin"] += 1
                continue
            if limit and len(rows) >= limit:
                break"""
assert s.count(old) == 1
s = s.replace(old, new)

# thread qmin through the loader's signature and its call site
import re
s = re.sub(r"def load_pairs\(([^)]*)\):", lambda m: "def load_pairs(%s, qmin=0.0):" % m.group(1),
           s, count=1)
n = len(re.findall(r"load_pairs\(", s))
s = re.sub(r"(load_pairs\((?!.*qmin)[^)]*)\)", r"\1, qmin=a.qmin)", s, count=0)
assert s != orig
open(p + ".tmp", "w").write(s)
os.replace(p + ".tmp", p)
print("patched dpo_teacher.py with --qmin (load_pairs occurrences: %d)" % n)
