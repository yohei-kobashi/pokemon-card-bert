#!/usr/bin/env python3
"""Step 1 (a) and (c) of the LoRA-split plan: is there any block structure to split ON?

Two signals, neither of which needs the GPU:

(a) BEHAVIOURAL. From lmlog_r*.jsonl.gz, which records every logged decision as
    (deck, offered kinds, picked kind, turn, prizes). Two distributions per deck:
      faces[k]  = P(a decision's menu contains kind k)      -- a property of the DECK
      picks[k]  = P(the pick is of kind k | the menu offered k)  -- a property of the PLAY
    `picks` is the one to cluster on. `faces` measures which situations a deck gets into;
    `picks` measures what the pilot does when it gets there, which is what a shared LoRA has
    to reconcile. Distance is Jensen-Shannon, which is bounded and symmetric.

(c) TRAJECTORY. From the per-deck win rates in the gate JSONs of successive rounds. If two
    decks rise and fall together under shared updates they are compatible; if one falls when
    the other rises the shared LoRA is trading them off. This is the most DIRECT evidence
    available but the weakest: four rounds and a gate that re-scores the same checkpoint 2.6pt
    apart. Reported for agreement with (a), never on its own.

Neither replaces the gradient measurement (Step 1b) -- (a) is about behaviour, not about
whether the updates conflict. But if BOTH come back structureless, there is nothing to split.
"""
import collections
import glob
import gzip
import json
import math
import os
import sys

DECKS = ["dragapult_dusknoir", "dragapult", "marnie_grimmsnarl", "alakazam_nz", "alakazam",
         "crustle_geco", "crustle", "ogerpon_mono", "dudunsparce_box", "cynthia_garchomp",
         "mega_lucario_tr", "slowking"]


def js(p, q):
    keys = set(p) | set(q)
    m = {k: 0.5 * (p.get(k, 0.0) + q.get(k, 0.0)) for k in keys}

    def kl(a):
        s = 0.0
        for k in keys:
            if a.get(k, 0.0) > 0 and m[k] > 0:
                s += a[k] * math.log(a[k] / m[k])
        return s
    return 0.5 * kl(p) + 0.5 * kl(q)


def norm(c):
    t = sum(c.values())
    return {k: v / t for k, v in c.items()} if t else {}


def behavioural(paths):
    faces = collections.defaultdict(collections.Counter)
    picks = collections.defaultdict(collections.Counter)
    n = collections.Counter()
    for p in paths:
        # Round 6 is collecting into these files right now, so the newest one has no end-of-
        # stream marker and gzip raises partway through. Everything read before the raise is
        # complete records and is kept: this is a distribution estimate, not a ledger.
        try:
            lines = list(gzip.open(p, "rt"))
        except EOFError:
            lines = []
            with gzip.open(p, "rt") as fh:
                try:
                    for ln in fh:
                        lines.append(ln)
                except EOFError:
                    pass
            print("  [partial] %s -> %d records (still being written)"
                  % (os.path.basename(p), len(lines)))
        for line in lines:
            try:
                d = json.loads(line)
            except Exception:
                continue
            if (d.get("n_cand") or 0) < 2:      # forced: no policy content
                continue
            deck = d.get("deck")
            if not deck:
                continue
            n[deck] += 1
            offered = {str(o).split(":", 1)[0].split("@", 1)[0] for o in (d.get("offered") or [])}
            for k in offered:
                faces[deck][k] += 1
            pk = str(d.get("pick_kind") or "?").split(":", 1)[0].split("@", 1)[0]
            picks[deck][pk] += 1
    return faces, picks, n


def trajectory(gate_globs):
    """-> {deck: [win rate per round]} in the order the globs are given."""
    series = collections.defaultdict(list)
    rounds = []
    for label, pat in gate_globs:
        decks = {}
        for p in sorted(glob.glob(pat)):
            try:
                j = json.load(open(p))
            except Exception:
                continue
            decks.update(j.get("decks", {}))
        if not decks:
            continue
        rounds.append(label)
        for d, v in decks.items():
            series[d].append((label, v.get("p")))
    return rounds, series


def main():
    logs = sorted(glob.glob("/root/lmlog_r*.jsonl.gz"))
    print("== (a) behavioural ==  %d log files" % len(logs))
    faces, picks, n = behavioural(logs)
    decks = [d for d in DECKS if n.get(d, 0) >= 500]
    print("decks with >=500 real decisions: %d of %d" % (len(decks), len(DECKS)))
    for d in DECKS:
        print("   %-22s %7d decisions%s" % (d, n.get(d, 0), "" if d in decks else "   [DROPPED]"))

    pv = {d: norm(picks[d]) for d in decks}
    kinds = sorted({k for d in decks for k in pv[d]}, key=lambda k: -sum(pv[d].get(k, 0) for d in decks))
    print("\npick-kind shares (%% of real decisions), most common first:")
    print("%-22s %s" % ("deck", " ".join("%8s" % k[:8] for k in kinds[:9])))
    for d in decks:
        print("%-22s %s" % (d, " ".join("%7.1f%%" % (100 * pv[d].get(k, 0)) for k in kinds[:9])))

    print("\nJensen-Shannon distance between pick distributions (x1000, lower = more alike):")
    print("%-22s %s" % ("", " ".join("%6s" % d[:6] for d in decks)))
    D = {}
    for a in decks:
        row = []
        for b in decks:
            v = js(pv[a], pv[b])
            D[(a, b)] = v
            row.append("%6.0f" % (1000 * v))
        print("%-22s %s" % (a, " ".join(row)))

    # Average linkage, printed as a merge order: the reader can cut it at k=2 or k=3 rather
    # than being handed one number of clusters as if it were a finding.
    print("\naverage-linkage merges (distance x1000):")
    groups = [[d] for d in decks]
    while len(groups) > 1:
        best = None
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                v = sum(D[(x, y)] for x in groups[i] for y in groups[j]) / (len(groups[i]) * len(groups[j]))
                if best is None or v < best[0]:
                    best = (v, i, j)
        v, i, j = best
        print("  %6.0f  {%s} + {%s}   -> %d groups left"
              % (1000 * v, ",".join(g[:12] for g in groups[i]), ",".join(g[:12] for g in groups[j]),
                 len(groups) - 1))
        groups[i] = groups[i] + groups[j]
        groups.pop(j)

    print("\n== (c) trajectory ==")
    rounds, series = trajectory([
        ("stage1_r1", "/root/loop_stage1/gate_r1.json"),
        ("stage1_r2", "/root/loop_stage1/gate_r2.json"),
        ("dpo_r4", "/root/loop_dpo/gate_dpo4.*.json"),
        ("dpo_r5", "/root/loop_dpo/gate_dpo5b.*.json"),
    ])
    print("rounds found: %s" % (rounds,))
    common = [d for d in DECKS if len(series.get(d, [])) == len(rounds) and rounds]
    print("decks present in every round: %d" % len(common))
    if len(rounds) >= 3 and len(common) >= 4:
        print("\n%-22s %s" % ("deck", " ".join("%10s" % r for r in rounds)))
        for d in common:
            print("%-22s %s" % (d, " ".join("%9.1f%%" % (100 * v) for _, v in series[d])))
        print("\nround-to-round deltas (pt):")
        dd = {}
        for d in common:
            xs = [v for _, v in series[d]]
            dd[d] = [100 * (xs[i + 1] - xs[i]) for i in range(len(xs) - 1)]
            print("%-22s %s" % (d, " ".join("%+7.1f" % x for x in dd[d])))
        print("\ncorrelation of those deltas (x100). %d points per pair -- read the SIGN only:"
              % len(next(iter(dd.values()))))
        print("%-22s %s" % ("", " ".join("%6s" % d[:6] for d in common)))
        for a in common:
            row = []
            for b in common:
                xa, xb = dd[a], dd[b]
                ma, mb = sum(xa) / len(xa), sum(xb) / len(xb)
                va = math.sqrt(sum((x - ma) ** 2 for x in xa))
                vb = math.sqrt(sum((x - mb) ** 2 for x in xb))
                c = (sum((x - ma) * (y - mb) for x, y in zip(xa, xb)) / (va * vb)) if va and vb else 0.0
                row.append("%6.0f" % (100 * c))
            print("%-22s %s" % (a, " ".join(row)))
    else:
        print("not enough rounds/decks for a trajectory read -- (a) stands alone until Step 1b")


if __name__ == "__main__":
    main()
