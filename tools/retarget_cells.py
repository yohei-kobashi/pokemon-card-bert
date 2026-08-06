#!/usr/bin/env python3
"""Re-weight the (deck, kind) Q-label budget by what tools/price_targets.py actually validated.

diag_lm_losses.py produces OBSERVED gaps: in a mirror game -- same decklist, same shuffle, same
policy -- the winning seat took kind K this much more or less often than the losing seat. That is
a controlled contrast but still a correlation, and its `share` field spends the branch budget as
if every gap were causal. price_targets.py answers the causal question one cell at a time. This
tool is the join: observed shares in, priced shares out, and nothing invented in between.

HEADROOM IS SIGN-TIMES-SIDE, NOT SIGN. price_targets.py fixes one axis -- positive dQ always
means "the flagged kind is undervalued here" -- but which sign indicates a MISTAKE depends on
which side was priced:

    decline side, dQ > 0    the LM declined and should have taken.  MISTAKE -> boost
    decline side, dQ < 0    the LM declined and was right.          settled on this side
    take side,    dQ < 0    the LM took and should not have.        MISTAKE -> boost
    take side,    dQ > 0    the LM took and was right.              settled on this side

Getting this wrong inverts the budget on exactly the cells the analysis cares most about: every
observed gap in this fleet is negative (winners take the kind LESS), so the live hypothesis is
over-taking, whose signature is a NEGATIVE take-side dQ -- the same sign that means "settled" on
the decline side. Measured: ogerpon_mono/end is -0.3825 on decline (right to decline) and -0.1346
at z -4.14 on take (wrong to take). It is a real target, and a sign-only rule would have cut its
budget to a third.

    priced, mistake, |z| >= 2   x BOOST
    priced, settled, |z| >= 2   x DEMOTE  -- on the side that was priced, and only that side
    priced, |z| < 2             x 1.0     measured and inconclusive; the observed prior stands
    not priced                  x 1.0     no evidence either way; the observed prior stands

DEMOTE, NOT DELETE: a cell settled on one side has never been branched on the other, and
qlabel_gen branches a decision whenever the kind is on the MENU, so those cells still generate
usable labels. They just stop out-bidding a cell with a measured mistake.

dQ_max from the take side is IGNORED here. It is biased downward by max-of-k selection and came
back significantly negative on all six cells including the ones every other measurement calls
positive -- see the note in price_targets.py.

SEAT PINS come from the same analysis, not from this tool's arithmetic. diag_lm_losses.py
section 6 found the Alakazam family collapsing by seat (34-6 and 32-8 for the first player) --
a different failure from a take-rate gap, and one that labels collected in the winning seat
cannot fix. --seat-pin deck:second pins those cells.

    python3 tools/retarget_cells.py --observed evaluations/lm_targets_i2r6.json \\
        --priced /root/priced_engine.json --priced /root/priced_take.json \\
        --seat-pin alakazam:second --seat-pin alakazam_nz:second \\
        --out evaluations/lm_targets_priced.json
"""

import argparse
import json
import math
import sys

BOOST, DEMOTE, FAMILY = 3.0, 0.35, 1.5


def pool_by_kind(priced, min_z):
    """Inverse-variance pool each (kind, side) across decks; return the kinds whose POOLED
    effect is a mistake.

    100 branch points per cell resolves ~0.03 in dQ, so a per-cell |z| >= 2 gate needs an effect
    near 0.06 and anything smaller reads as "settled" one deck at a time. But there is one model
    conditioned on the deck through DECK[]/ID ME, not eleven models, so a kind that leans the
    same way on every deck is one property of that model and the right unit of evidence is the
    pooled estimate. Measured on the take side: end came back -0.135, -0.061, -0.052, -0.037 --
    four negatives, one of which clears |z| 2 alone. Pooled it is z -4.3. Per-cell gating would
    have given ogerpon_mono/end 15.7% and cut the other three to ~1.3% each, which is a
    statement about which cell got lucky at n=100, not about the policy.
    """
    agg = {}
    for (deck, kind), rows in priced.items():
        for r in rows:
            se = r.get("se") or 0.0
            if se <= 0:
                continue
            k = (kind, r.get("side", "decline"))
            w = 1.0 / (se * se)
            s = agg.setdefault(k, [0.0, 0.0, 0])
            s[0] += w * r["dQ"]
            s[1] += w
            s[2] += 1
    out = {}
    for (kind, side), (num, w, n) in agg.items():
        if n < 2 or w <= 0:
            continue
        m, se = num / w, 1.0 / math.sqrt(w)
        z = m / se
        if abs(z) >= min_z and (m > 0) == (side == "decline"):
            out[kind] = (side, m, z, n)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--observed", required=True, help="diag_lm_losses.py --targets output")
    ap.add_argument("--priced", action="append", default=[],
                    help="price_targets.py output; repeatable (e.g. decline and take runs)")
    ap.add_argument("--seat-pin", action="append", default=[], metavar="DECK:SEAT")
    ap.add_argument("--min-z", type=float, default=2.0)
    ap.add_argument("--no-pool-kinds", dest="pool_kinds", action="store_false",
                    help="judge every cell on its own |z| only (see pool_by_kind for why not)")
    ap.add_argument("--floor", type=float, default=0.01, help="share floor per surviving cell")
    # A single validated cell can otherwise take ~44% of the budget, which is the exact shape
    # [[narrow-dagger-overfits]] measured: DAgger concentrated on one deck moved that deck
    # +11.9pt and the fleet -2.75pt. The pricing says where the headroom IS; it does not say
    # that spending the whole round there is safe.
    ap.add_argument("--max-share", type=float, default=0.25, help="cap on any single cell")
    ap.add_argument("--drop-below", type=float, default=0.0,
                    help="drop cells whose final share is under this (0 = keep all)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    obs = json.load(open(a.observed))
    cells = obs["cells"]
    pins = dict(x.split(":", 1) for x in a.seat_pin)

    # A cell can be priced on both sides. Keep every row: a positive on either side is evidence
    # of headroom, so the boost must not be overwritten by a later demotion from the other side.
    priced = {}
    for p in a.priced:
        try:
            rows = json.load(open(p))["cells"]
        except Exception as e:
            sys.exit("cannot read %s: %s" % (p, e))
        for r in rows:
            priced.setdefault((r["deck"], r["kind"]), []).append(r)

    fam = pool_by_kind(priced, a.min_z) if a.pool_kinds else {}
    for kind, (side, m, z, n) in sorted(fam.items()):
        print("[pooled] %-8s %-7s %+.4f  z %+5.2f over %d decks -> every %s cell floors at x%.1f"
              % (kind, side, m, z, n, kind, FAMILY))

    out, notes = [], []
    for c in cells:
        key = (c["deck"], c["kind"])
        rows = priced.get(key, [])
        sig = [r for r in rows if abs(r.get("z", 0.0)) >= a.min_z]
        # a mistake is dQ > 0 when the LM declined, dQ < 0 when it took
        mistakes = [r for r in sig
                    if (r["dQ"] > 0) == (r.get("side", "decline") == "decline")]
        if mistakes:
            mult = BOOST
            why = "MISTAKE (%s)" % "/".join("%s %+.3f" % (r.get("side", "decline"), r["dQ"])
                                            for r in mistakes)
        elif sig:
            mult = DEMOTE
            why = "settled (%s)" % "/".join(r.get("side", "decline") for r in sig)
        elif rows:
            mult, why = 1.0, "priced, |z| < %.1f" % a.min_z
        else:
            mult, why = 1.0, "unpriced"
        # The pooled kind effect is a FLOOR, never a ceiling: a cell that individually cleared
        # the gate keeps its full boost, and a cell the pooling rescues is not demoted below it.
        if c["kind"] in fam and mult < FAMILY:
            mult, why = FAMILY, "pooled %s (%s)" % (c["kind"], fam[c["kind"]][0])
        d = dict(c)
        d["share_raw"] = c.get("share", 0.0) * mult
        d["priced"] = why
        d["dQ"] = round(mistakes[0]["dQ"], 4) if mistakes else None
        if c["deck"] in pins:
            d["seat"] = pins[c["deck"]]
        out.append(d)
        notes.append((d["share_raw"], c["deck"], c["kind"], why, mult))

    tot = sum(d["share_raw"] for d in out) or 1.0
    for d in out:
        d["share"] = round(max(a.floor, d["share_raw"] / tot), 4)
        d.pop("share_raw")
    # Cap, then give the spill back to everything under the cap -- renormalising all cells would
    # just push the capped one back over it.
    for _ in range(8):
        over = [d for d in out if d["share"] > a.max_share]
        if not over:
            break
        spill = sum(d["share"] - a.max_share for d in over)
        for d in over:
            d["share"] = a.max_share
        rest = [d for d in out if d["share"] < a.max_share]
        base = sum(d["share"] for d in rest) or 1.0
        for d in rest:
            d["share"] += spill * d["share"] / base
    tot2 = sum(d["share"] for d in out)
    for d in out:
        d["share"] = round(d["share"] / tot2, 4)
    if a.drop_below:
        out = [d for d in out if d["share"] >= a.drop_below]

    out.sort(key=lambda d: -d["share"])
    json.dump({"min_z": obs.get("min_z"), "priced_from": a.priced,
               "seat_pins": pins, "cells": out}, open(a.out, "w"), indent=1)

    print("%-20s %-8s %-6s %7s -> %7s   %s" % ("deck", "kind", "seat", "obs", "new", "why"))
    prev_share = {(c["deck"], c["kind"]): c.get("share", 0.0) for c in cells}
    for d in out:
        print("%-20s %-8s %-6s %6.1f%% -> %6.1f%%   %s"
              % (d["deck"], d["kind"], d.get("seat", "any"),
                 100 * prev_share[(d["deck"], d["kind"])], 100 * d["share"], d["priced"]))
    print("\nwrote %d cells to %s" % (len(out), a.out))


if __name__ == "__main__":
    main()
