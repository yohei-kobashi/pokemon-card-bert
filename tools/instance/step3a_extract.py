#!/usr/bin/env python3
"""Extract branch points that carry playout Q, for teacher scoring on the other instance.

Filters are rank_probe's, so the numbers stay comparable to the policy's measured +0.0469:
at least 2 candidates with a Q, and at least 2 DISTINCT Q values (otherwise there is nothing
to rank and the metric has no signal).
"""
import glob
import gzip
import json
import sys

pats, out_path, want = sys.argv[1:-2], sys.argv[-2], int(sys.argv[-1])

rows = []
for pat in pats:
    for path in sorted(glob.glob(pat)):
        for line in gzip.open(path, "rt"):
            d = json.loads(line)
            q = d.get("qvals")
            if not q:
                continue
            idx = [i for i, v in enumerate(q) if v is not None]
            if len(idx) < 2:
                continue
            cands = d.get("cands") or []
            if len(cands) != len(q):
                continue
            if len({q[i] for i in idx}) < 2:
                continue
            rows.append({"prompt": d["prompt"], "cands": cands, "idx": idx,
                         "q": [q[i] for i in idx], "matchup": d.get("matchup", "")})

# deterministic stride, so the sample spreads over every file and matchup rather than taking
# the first N (which would be one matchup's opening turns)
step = max(1, len(rows) // want)
sample = rows[::step][:want]
with gzip.open(out_path, "wt") as f:
    for r in sample:
        f.write(json.dumps(r) + "\n")

ks = [len(r["idx"]) for r in sample]
print("eligible %d -> sampled %d (stride %d)" % (len(rows), len(sample), step))
print("scorable candidates/decision: mean %.2f min %d max %d"
      % (sum(ks) / len(ks), min(ks), max(ks)))
print("matchups %d" % len({r["matchup"] for r in sample}))
print("\n--- first prompt (check the FORMAT matches what the teacher was trained on) ---")
print(sample[0]["prompt"][:400])
