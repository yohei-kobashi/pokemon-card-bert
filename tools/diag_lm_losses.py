#!/usr/bin/env python3
"""What separates the LM's wins from its losses -- with engine_v2 nowhere in the measurement.

Reads tools/lm_mirror_log.py output. Both seats of every game are the SAME model on the SAME
decklist with the SAME shuffle, so within one game the only differences are turn order and the
moves chosen. That makes a within-seat win/loss comparison a controlled contrast rather than a
correlation hunt: draw luck is identical by construction.

THREE STATISTICS, IN INCREASING ORDER OF HOW MUCH YOU SHOULD BELIEVE THEM.

1. SEAT SPLIT. What share of games the first player wins. This is not a finding, it is the
   size of the confound everything else has to be protected from. Reported first so a large
   value cannot be quietly absorbed into the other numbers.

2. TAKE-RATE GAP, menu-conditioned. For each action kind: among decisions where that kind was
   OFFERED, how often was it TAKEN, in games this seat won vs games this seat lost. Conditioning
   on the offer is what separates "the loser retreats less" from "the loser was offered fewer
   retreats" -- the pooling trap that inverted the trend in [[systematic-divergence-diagnostic]].

3. EARLY, PRIZE-MATCHED TAKE-RATE GAP. The same thing restricted to turns <= --early-turn and
   to decisions where the two prize counts are EQUAL. This is the one to act on.

   Why: splitting on the outcome conditions on the future, so a raw gap can be an effect rather
   than a cause -- the winner attacks more BECAUSE it is ahead. Restricting to even prizes and
   to early turns removes the states where the outcome is already largely decided.
   `setup-execution-audit-and-budew-overattack` is the burned precedent for skipping this step:
   over-attack was a symptom, and the generic rule built on it helped feraligatr and regressed
   dragapult.

   A kind whose gap survives (3) is a candidate CAUSE. A kind that only shows up in (2) is a
   symptom and must not be turned into a rule.

Nothing here proves value -- a surviving gap is a hypothesis to be priced, not a fix. The
pricing step branches only those kinds, which is what makes it cheap enough to run with the LM
itself as the rollout policy instead of engine_v2.

    python3 tools/diag_lm_losses.py /root/lmlog_r7.jsonl.gz --top 10
    python3 tools/diag_lm_losses.py /root/lmlog_r7.jsonl.gz --deck dragapult --detail
"""

import argparse
import collections
import gzip
import json
import math
import sys


def wilson_gap(k1, n1, k0, n0):
    """Difference of two rates with a normal SE. Returns (gap_pp, se_pp, z)."""
    if n1 < 1 or n0 < 1:
        return 0.0, 0.0, 0.0
    p1, p0 = k1 / n1, k0 / n0
    se = math.sqrt(max(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0, 1e-12))
    return 100 * (p1 - p0), 100 * se, (p1 - p0) / se


def tally(rows, kinds):
    """-> {kind: [taken_won, offered_won, taken_lost, offered_lost]}"""
    t = {k: [0, 0, 0, 0] for k in kinds}
    for r in rows:
        w = r["won"]
        pk = r["pick_kind"]
        for k in r["offered"]:
            if k not in t:
                continue
            t[k][1 if w else 3] += 1
            if pk == k:
                t[k][0 if w else 2] += 1
    return t


def _group_gaps(per_game, kinds, wseat):
    """Per-game (winner take-rate - loser take-rate) per kind, for games won by `wseat`."""
    diffs = collections.defaultdict(list)
    for _key, sides in per_game.items():
        if not sides[0] or not sides[1] or sides[wseat][0]["won"] != 1:
            continue
        win_rows, lose_rows = sides[wseat], sides[1 - wseat]
        for k in kinds:
            ow = sum(1 for r in win_rows if k in r["offered"])
            ol = sum(1 for r in lose_rows if k in r["offered"])
            if ow < 3 or ol < 3:
                continue
            tw = sum(1 for r in win_rows if k in r["offered"] and r["pick_kind"] == k)
            tl = sum(1 for r in lose_rows if k in r["offered"] and r["pick_kind"] == k)
            diffs[k].append(tw / ow - tl / ol)
    return diffs


def _mean_se(xs):
    if len(xs) < 2:
        return (xs[0] if xs else 0.0), 0.0
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))
    return m, sd / math.sqrt(len(xs))


def per_kind_combined(per_game, kinds, top, verbose=True, min_games=8):
    """Equal-weight mean of the two winner-seat groups -> the seat bias cancels exactly."""
    g0, g1 = _group_gaps(per_game, kinds, 0), _group_gaps(per_game, kinds, 1)
    out, res = [], {}
    for k in kinds:
        a0, a1 = g0.get(k, []), g1.get(k, [])
        if len(a0) < min_games or len(a1) < min_games:
            continue
        m0, s0 = _mean_se(a0)
        m1, s1 = _mean_se(a1)
        m = (m0 + m1) / 2.0
        se = math.sqrt(s0 ** 2 + s1 ** 2) / 2.0
        res[k] = (m, se)
        out.append((abs(m / se) if se else 0.0, m, se, k, m0, m1, len(a0) + len(a1)))
    if verbose:
        out.sort(reverse=True)
        print("   %-10s %10s %8s %9s %9s %7s" % ("kind", "gap", "z", "seat0-win", "seat1-win", "games"))
        for z, m, se, k, m0, m1, n in (out[:top] if top else out):
            print("   %-10s %+8.1fpp ±%.1f %+7.2f %+8.1f %+9.1f %6d"
                  % (k, 100 * m, 100 * se, (m / se) if se else 0.0, 100 * m0, 100 * m1, n))
    return res


def report(name, rows, kinds, top, min_offered):
    t = tally(rows, kinds)
    out = []
    for k, (tw, ow, tl, ol) in t.items():
        if min(ow, ol) < min_offered:
            continue
        gap, se, z = wilson_gap(tw, ow, tl, ol)
        out.append((abs(z), gap, se, z, k, ow, ol, 100 * tw / max(1, ow), 100 * tl / max(1, ol)))
    out.sort(reverse=True)
    print("\n%s   (%d decisions)" % (name, len(rows)))
    if not out:
        print("   (no action kind offered >= %d times on both sides)" % min_offered)
        return []
    print("   %-10s %8s %8s %9s %7s %7s" % ("kind", "won%", "lost%", "gap", "z", "offered"))
    for _, gap, se, z, k, ow, ol, pw, pl in out[:top]:
        print("   %-10s %7.1f%% %7.1f%%  %+6.1f%s %+6.2f %6d/%d"
              % (k, pw, pl, gap, "±%.1f" % se, z, ow, ol))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--deck", default="", help="comma list; default = all, pooled + per deck")
    ap.add_argument("--early-turn", type=int, default=8)
    ap.add_argument("--min-offered", type=int, default=150)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--detail", action="store_true", help="per-deck sections for every deck")
    ap.add_argument("--targets", default="",
                    help="write a (deck, kind) branch-budget allocation here as JSON")
    ap.add_argument("--min-z", type=float, default=1.5,
                    help="--targets: keep a (deck, kind) cell at or above this |z|")
    a = ap.parse_args()

    rows = []
    for f in a.files:
        op = gzip.open if f.endswith(".gz") else open
        with op(f, "rt") as fh:
            rows += [json.loads(l) for l in fh]
    if a.deck:
        want = set(a.deck.split(","))
        rows = [r for r in rows if r["deck"] in want]
    if not rows:
        sys.exit("no rows")

    kinds = sorted({k for r in rows for k in r["offered"]})
    decks = sorted({r["deck"] for r in rows})
    print("%d decisions | %d decks | %d action kinds | kinds: %s"
          % (len(rows), len(decks), len(kinds), " ".join(kinds)))

    # ---- 1. the confound, first -------------------------------------------------------------
    games = {}
    for r in rows:
        games[(r["deck"], r["seed"])] = games.get((r["deck"], r["seed"]), None)
        if r["won"]:
            games[(r["deck"], r["seed"])] = r["seat"]
    seat_w = collections.Counter(v for v in games.values() if v is not None)
    ng = sum(seat_w.values())
    print("\n=== 1. SEAT SPLIT (the confound, not a finding) ===")
    if ng:
        p = seat_w[0] / ng
        se = math.sqrt(p * (1 - p) / ng)
        print("   first player (seat 0) wins %d/%d = %.1f%% ± %.1f"
              % (seat_w[0], ng, 100 * p, 100 * se))
        if abs(p - 0.5) > 3 * se:
            print("   -> significant. Every take-rate below is computed WITHIN a seat for this"
                  " reason;\n      never pool the seats.")
    else:
        print("   no decided games")

    # ---- 2 and 3, within each seat ----------------------------------------------------------
    for seat in (0, 1):
        sr = [r for r in rows if r["seat"] == seat]
        if not sr:
            continue
        print("\n=== 2. TAKE-RATE GAP, menu-conditioned | SEAT %d ===" % seat)
        report("all decisions", sr, kinds, a.top, a.min_offered)

        early = [r for r in sr
                 if (r.get("turn") or 0) <= a.early_turn and r["my_pz"] == r["op_pz"]]
        print("\n=== 3. EARLY (turn<=%d) + PRIZE-MATCHED | SEAT %d   <- act on this one ==="
              % (a.early_turn, seat))
        report("early, even prizes", early, kinds, a.top, max(30, a.min_offered // 4))

    # ---- per deck ---------------------------------------------------------------------------
    if a.detail or len(decks) == 1:
        for d in decks:
            dr = [r for r in rows if r["deck"] == d
                  and (r.get("turn") or 0) <= a.early_turn and r["my_pz"] == r["op_pz"]]
            if len(dr) < 200:
                continue
            report("DECK %s (early, even prizes, seats pooled)" % d, dr, kinds, 5, 30)
        print("\n   NOTE: per-deck sections pool the seats to keep the counts usable. Read them"
              "\n   as a pointer to which deck carries a fleet-level gap, not as evidence on"
              "\n   their own.")

    # ---- 4. WITHIN-GAME paired winner vs loser ----------------------------------------------
    # THE statistic mirror was paid for, and the only one where resources are exactly equal.
    #
    # Sections 2 and 3 compare a seat's WINS against its LOSSES, which are different seeds and
    # therefore different shuffles -- draw luck is back in as a confound. Inside ONE mirror
    # game both seats hold the same decklist in the same order, so they draw the same cards in
    # the same sequence; the only differences are turn order and the moves chosen. Pairing
    # within the game removes the shuffle entirely.
    #
    # Turn order is handled by reporting the two winner-seat groups SEPARATELY. A gap that
    # flips sign between them is a first-player artifact; one that holds in both is not.
    print("\n=== 4. WITHIN-GAME: winner vs loser, SAME shuffle   <- the strongest statistic ===")
    per_game = collections.defaultdict(lambda: {0: [], 1: []})
    for r in rows:
        per_game[(r["deck"], r["seed"])][r["seat"]].append(r)

    for wseat in (0, 1):
        # paired per-game difference in take-rate, per kind
        diffs = collections.defaultdict(list)
        ngame = 0
        for (deck, seed), sides in per_game.items():
            if not sides[0] or not sides[1]:
                continue
            if sides[wseat] and sides[wseat][0]["won"] != 1:
                continue
            ngame += 1
            win_rows, lose_rows = sides[wseat], sides[1 - wseat]
            for k in kinds:
                ow = sum(1 for r in win_rows if k in r["offered"])
                ol = sum(1 for r in lose_rows if k in r["offered"])
                if ow < 3 or ol < 3:          # a rate from 1-2 offers is noise, not a rate
                    continue
                tw = sum(1 for r in win_rows if k in r["offered"] and r["pick_kind"] == k)
                tl = sum(1 for r in lose_rows if k in r["offered"] and r["pick_kind"] == k)
                diffs[k].append(tw / ow - tl / ol)
        print("\n  winner was seat %d   (%d games)" % (wseat, ngame))
        if not ngame:
            continue
        out = []
        for k, ds in diffs.items():
            if len(ds) < 20:
                continue
            m = sum(ds) / len(ds)
            sd = math.sqrt(sum((x - m) ** 2 for x in ds) / max(1, len(ds) - 1))
            se = sd / math.sqrt(len(ds))
            out.append((abs(m / se) if se else 0, m, se, k, len(ds)))
        out.sort(reverse=True)
        print("   %-10s %10s %8s %8s" % ("kind", "gap", "z", "games"))
        for z, m, se, k, n in out[:a.top]:
            # se == 0 means every game gave the identical difference (usually a kind that is
            # always taken when offered, e.g. a forced `yes`). Print it rather than divide.
            print("   %-10s %+8.1fpp ±%.1f %7s %6d"
                  % (k, 100 * m, 100 * se, ("%+.2f" % (m / se)) if se else "--", n))

    # ---- 4b. cancel the seat bias by averaging the two winner-seat groups -------------------
    # The turn-order effect enters the within-game gap ADDITIVELY and with OPPOSITE SIGN in the
    # two groups: seat 0 attaches more whether or not it wins, so "winner attaches more" when
    # seat 0 wins and "winner attaches less" when seat 1 wins. Writing
    #     gap = effect + bias * s,   s = +1 if the winner was seat 0, -1 otherwise
    # the EQUAL-WEIGHT mean of the two group means is `effect` exactly. Equal weight matters --
    # pooling the rows instead would weight by group size and leave the bias behind.
    #
    # The check that this is not wishful: `attach` cancels to 0.0pp. A kind whose entire signal
    # is turn order lands on zero, which is what a working correction looks like.
    print("\n=== 4b. SEAT-BIAS CANCELLED (mean of the two groups) -- the headline number ===")
    combined = per_kind_combined(per_game, kinds, a.top, verbose=True)

    if a.detail or len(decks) > 1:
        print("\n=== 5. PER DECK, seat-bias cancelled ===")
        print("   %-20s %6s %7s %8s %8s %8s %8s"
              % ("deck", "games", "seat0%", "end", "play", "retreat", "evolve"))
        watch = ["end", "play", "retreat", "evolve"]
        for d in decks:
            pg = {k: v for k, v in per_game.items() if k[0] == d}
            ng = sum(1 for v in pg.values() if v[0] and v[1])
            s0 = sum(1 for v in pg.values()
                     if v[0] and v[1] and v[0][0]["won"] == 1)
            c = per_kind_combined(pg, kinds, 0, verbose=False)
            cells = []
            for k in watch:
                if k in c:
                    m, se = c[k]
                    star = "*" if se and abs(m / se) >= 2 else " "
                    cells.append("%+6.1f%s" % (100 * m, star))
                else:
                    cells.append("    --")
            print("   %-20s %6d %6.0f%% %8s %8s %8s %8s"
                  % (d, ng, 100.0 * s0 / max(1, ng), *cells))
        print("   (* = |z| >= 2 within that deck; ~40 games per deck, so read the SIGN and the"
              "\n    fleet consistency, not one deck's magnitude)")

    # ---- 6. decks that cannot be measured at all, which is itself the finding ---------------
    # A deck whose first-player win rate is extreme has almost no games won from the other
    # seat, so the seat-cancelled statistic has nothing to average and prints "--". Silence
    # there is not "no problem" -- it IS the problem, and it is a different one: the pilot
    # cannot win from that seat at all. Surfaced separately so it never reads as missing data.
    print("\n=== 6. SEAT COLLAPSE (a take-rate gap is not the issue here) ===")
    any_collapse = False
    for d in decks:
        pg = {k: v for k, v in per_game.items() if k[0] == d}
        n0 = sum(1 for v in pg.values() if v[0] and v[1] and v[0][0]["won"] == 1)
        n1 = sum(1 for v in pg.values() if v[0] and v[1] and v[1][0]["won"] == 1)
        if min(n0, n1) < 10 and (n0 + n1) >= 20:
            any_collapse = True
            bad = 1 if n1 < n0 else 0
            print("   %-20s seat0 %d - seat1 %d wins  -> cannot win from seat %d"
                  % (d, n0, n1, bad))
    if not any_collapse:
        print("   (none)")
    else:
        print("   These are TRAINING targets like any other: the LM is one model conditioned on"
              "\n   the deck, so a per-deck weakness is fixed by training on that deck, not by"
              "\n   writing engine config. Give them pilot mass in the seat they lose from.")

    # ---- targets file -----------------------------------------------------------------------
    if a.targets:
        tg = []
        for d in decks:
            pg = {k: v for k, v in per_game.items() if k[0] == d}
            for k, (m, se) in per_kind_combined(pg, kinds, 0, verbose=False).items():
                if se and abs(m / se) >= a.min_z and m != 0.0:
                    tg.append({"deck": d, "kind": k, "gap_pp": round(100 * m, 2),
                               "se_pp": round(100 * se, 2), "z": round(m / se, 2)})
        tg.sort(key=lambda x: -abs(x["z"]))
        # branch budget proportional to |gap| * sqrt(n-ish); |gap| alone is the honest weight
        tot = sum(abs(x["gap_pp"]) for x in tg) or 1.0
        for x in tg:
            x["share"] = round(abs(x["gap_pp"]) / tot, 4)
        with open(a.targets, "w") as f:
            json.dump({"min_z": a.min_z, "cells": tg}, f, indent=1)
        print("\nwrote %d (deck, kind) cells to %s  |  top: %s"
              % (len(tg), a.targets,
                 ", ".join("%s/%s %+.1f" % (x["deck"], x["kind"], x["gap_pp"]) for x in tg[:5])))

    print("\nHOW TO READ THIS. Section 4 is the controlled contrast: same deck, same shuffle,"
          "\nsame policy, one seat won. A kind whose gap holds with the SAME SIGN in both"
          "\nwinner-seat groups is a candidate CAUSE worth pricing by branching."
          "\nSections 2-3 compare different games (different shuffles), so a gap there that"
          "\ndoes NOT appear in section 4 is draw luck or an outcome artifact, not a lever."
          "\nNothing here proves value: a surviving gap is a hypothesis, and"
          "\n[[setup-execution-audit-and-budew-overattack]] is what acting on one without"
          "\npricing it looks like.")


if __name__ == "__main__":
    main()
