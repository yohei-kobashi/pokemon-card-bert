#!/usr/bin/env python3
"""Find SYSTEMATIC divergences between the LM and engine_v2, from DAgger rows already on disk.

Why this and not per-decision adjudication. Counterfactual playouts
([[dagger-label-is-a-coin-flip]]) continue with engine_v2 on BOTH sides, so they answer "does
this ONE move matter if the rest is played well" -- and a failure that repeats every turn is
invisible to that by construction: fix one instance, let engine_v2 rescue the rest, and the cost
disappears. ns_zoroark goes 0-40 in the mirror screen while no single decision measures as
decisive (mean Q gap -0.0003). That combination IS the signature of a systematic failure, and it
needs a statistic over whole games rather than over decisions.

Each DAgger row already carries the candidate menu, engine_v2's pick, and the LM's pick at the
same state, so the comparison is exactly paired -- no replay, no GPU, no new games.

Two statistics, because the raw one lies:

  pick share      P(kind | picked). Confounded: a kind picked often may simply be offered often.
  TAKE RATE       P(picked kind | kind was offered). This is the one to read -- it holds the menu
                  fixed, so a gap is a preference difference and not a menu difference.

    python3 tools/diag_systematic.py /root/loop_rerank6/dagger_r4.jsonl.gz
    python3 tools/diag_systematic.py --deck ns_zoroark --detail  <files...>
"""

import argparse
import collections
import glob
import gzip
import json
import math
import sys


def kind(cand):
    """Action kind from lm.actions.encode_option's `kind:detail@target` grammar."""
    if not isinstance(cand, str):
        return "?"
    return cand.split(":", 1)[0].split("@", 1)[0] or "?"


def load(paths):
    rows = []
    for pat in paths:
        for p in sorted(glob.glob(pat)) or [pat]:
            op = gzip.open if p.endswith(".gz") else open
            got = 0
            try:
                with op(p, "rt") as f:
                    for line in f:
                        try:
                            rows.append(json.loads(line))
                            got += 1
                        except Exception:
                            pass
            except (OSError, EOFError) as e:
                # A round still collecting has a half-written .gz. Keep what was readable and
                # say so, rather than failing the whole run -- looking at the current round
                # while it fills is exactly when this tool is most useful.
                print("%s: truncated, using the %d rows read (%s)"
                      % (p.split("/")[-1], got, type(e).__name__), file=sys.stderr)
    return rows


def tally(rows):
    """-> (offered, eng_take, lm_take, eng_pick, lm_pick, n) counters keyed by kind."""
    offered = collections.Counter()
    eng_take, lm_take = collections.Counter(), collections.Counter()
    eng_pick, lm_pick = collections.Counter(), collections.Counter()
    n = 0
    for r in rows:
        cands = r.get("candidates")
        if not cands:
            continue
        ci, li = r.get("chosen"), r.get("lm_chosen")
        if ci is None or not (0 <= ci < len(cands)):
            continue
        n += 1
        ks = {kind(c) for c in cands}
        for k in ks:
            offered[k] += 1
        ek = kind(cands[ci])
        eng_take[ek] += 1
        eng_pick[ek] += 1
        if li is not None and 0 <= li < len(cands):
            lk = kind(cands[li])
            lm_take[lk] += 1
            lm_pick[lk] += 1
    return offered, eng_take, lm_take, eng_pick, lm_pick, n


def tvd(a, b):
    """Total variation distance between two pick distributions, in [0,1]."""
    ta, tb = sum(a.values()) or 1, sum(b.values()) or 1
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a[k] / ta - b[k] / tb) for k in keys)


def detail(name, rows, min_offered, top):
    offered, eng_take, lm_take, eng_pick, lm_pick, n = tally(rows)
    if not n:
        return None
    d = tvd(eng_pick, lm_pick)
    ks = [k for k in offered if offered[k] >= min_offered]
    # Delta in TAKE RATE, with the SE of a difference of two proportions on the same denominator.
    scored = []
    for k in ks:
        m = offered[k]
        pe, pl = eng_take[k] / m, lm_take[k] / m
        se = math.sqrt(max(1e-12, (pe * (1 - pe) + pl * (1 - pl)) / m))
        scored.append((pl - pe, k, m, pe, pl, (pl - pe) / se if se else 0.0))
    scored.sort(key=lambda x: -abs(x[0]))
    if top:
        print("\n  %s  (n=%d rows, TVD %.3f)" % (name, n, d))
        print("    %-14s%9s%11s%11s%10s%8s" % ("kind", "offered", "engine%", "LM%", "delta", "z"))
        for delta, k, m, pe, pl, z in scored[:top]:
            flag = "  <-- over" if delta > 0 else "  <-- under"
            print("    %-14s%9d%10.1f%%%10.1f%%%+9.1fpt%8.1f%s"
                  % (k, m, 100 * pe, 100 * pl, 100 * delta, z, flag if abs(z) > 4 else ""))
    return n, d, scored


def confusion(rows, top):
    """What the LM substitutes for what: engine's kind -> the LM's kind, row-normalised."""
    cm = collections.Counter()
    tot = collections.Counter()
    for r in rows:
        cands = r.get("candidates")
        ci, li = r.get("chosen"), r.get("lm_chosen")
        if not cands or ci is None or li is None:
            continue
        if not (0 <= ci < len(cands) and 0 <= li < len(cands)):
            continue
        ek, lk = kind(cands[ci]), kind(cands[li])
        if ek == lk:
            continue                      # same kind: not a kind-level substitution
        cm[(ek, lk)] += 1
        tot[ek] += 1
    print("\n  substitutions (engine's kind -> what the LM played instead), kind-level only")
    for (ek, lk), c in cm.most_common(top):
        print("    %-12s -> %-12s %6d   (%.0f%% of engine's %s that the LM changed)"
              % (ek, lk, c, 100.0 * c / max(1, tot[ek]), ek))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--deck", default="", help="comma list; default = fleet view + worst decks")
    ap.add_argument("--min-offered", type=int, default=200)
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--worst", type=int, default=5, help="per-deck detail for the N worst")
    ap.add_argument("--detail", action="store_true", help="detail for every selected deck")
    ap.add_argument("--prize-band", default="", help="5-6 | 2-4 | 0-1 (needs the `prizes` field)")
    ap.add_argument("--track", default="",
                    help="treat each FILE as one round and table the take-rate gap per kind "
                         "across them, for this deck (or FLEET to pool every deck). The point "
                         "of the tool: a gap that does not shrink round over round is a "
                         "systematic failure the loop is not fixing.")
    a = ap.parse_args()

    if a.track:
        cols, per = [], {}
        loaded = [(f, load([f])) for f in a.files]
        common = None
        if a.track == "FLEET":
            # The target list is re-chosen every round, so pooling every deck compares different
            # deck mixes and the "trend" is composition. Restrict to the decks present in ALL
            # rounds; that is the only pooled comparison that means anything.
            for _, rs in loaded:
                ds = {r.get("deck") for r in rs}
                common = ds if common is None else (common & ds)
            common.discard(None)
            print("FLEET restricted to the %d deck(s) present in every round: %s\n"
                  % (len(common), ", ".join(sorted(common)) or "(none)"))
            if not common:
                sys.exit("no deck appears in every round -- track a single deck instead")
        for f, rs in loaded:
            if a.track != "FLEET":
                rs = [r for r in rs if r.get("deck") == a.track]
            else:
                rs = [r for r in rs if r.get("deck") in common]
            if not rs:
                print("%-28s (no rows)" % f.split("/")[-1]); continue
            offered, eng, lm, _, _, n = tally(rs)
            cols.append((f.split("/")[-1], n))
            for k, m in offered.items():
                if m >= a.min_offered:
                    per.setdefault(k, {})[cols[-1][0]] = (lm[k] - eng[k]) / m
        if not cols:
            sys.exit("nothing to track")
        print("take-rate gap (LM - engine), %s\n" % a.track)
        print("  %-12s" % "kind" + "".join("%16s" % c[:16] for c, _ in cols))
        print("  %-12s" % "(rows)" + "".join("%16d" % n for _, n in cols))
        order = sorted(per, key=lambda k: -max(abs(v) for v in per[k].values()))
        for k in order:
            cells = "".join(("%+15.1fpt" % (100 * per[k][c]) if c in per[k] else "%16s" % "-")
                            for c, _ in cols)
            print("  %-12s%s" % (k, cells))
        return

    rows = load(a.files)
    if a.prize_band:
        lo, hi = {"5-6": (5, 9), "2-4": (2, 4), "0-1": (0, 1)}[a.prize_band]
        before = len(rows)
        rows = [r for r in rows if "prizes" in r and lo <= r["prizes"] <= hi]
        print("prize band %s: %d of %d rows (rows without a `prizes` field are dropped)"
              % (a.prize_band, len(rows), before))
    if not rows:
        sys.exit("no rows")
    print("loaded %d rows from %d file pattern(s)" % (len(rows), len(a.files)))

    by = collections.defaultdict(list)
    for r in rows:
        by[r.get("deck", "?")].append(r)

    want = [d.strip() for d in a.deck.split(",") if d.strip()]
    print("\n=== fleet: decks ranked by how differently the LM picks ===")
    print("  %-24s%8s%8s   %s" % ("deck", "rows", "TVD", "biggest take-rate gaps"))
    ranked = []
    for d in sorted(by):
        got = detail(d, by[d], a.min_offered, 0)
        if not got:
            continue
        n, tv, scored = got
        tips = ", ".join("%s %+.0fpt" % (k, 100 * dl) for dl, k, *_ in scored[:3])
        ranked.append((tv, d, n, tips))
    ranked.sort(reverse=True)
    for tv, d, n, tips in ranked:
        print("  %-24s%8d%8.3f   %s" % (d, n, tv, tips))

    picked = want or [d for _, d, _, _ in ranked[:a.worst]]
    print("\n=== per-deck detail ===")
    for d in picked:
        if d in by:
            detail(d, by[d], a.min_offered, a.top)
            if a.detail or len(picked) <= 3:
                confusion(by[d], 6)


if __name__ == "__main__":
    main()
