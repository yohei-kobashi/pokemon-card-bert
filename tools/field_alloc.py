#!/usr/bin/env python3
"""Spend the round's game budget where the champion is losing, not evenly.

    tools/field_alloc.py --gate /root/loop_dusk/field/gate_r4.json --total 800
    -> marnie_grimmsnarl=101,alakazam_nz=113,dragapult=78,...      (sums to --total)

WHY. field_chain collects COLLECT games against each of eight opponents, and the pairs that
reach training come out almost perfectly even -- measured on round 4: 11.3% to 14.3% per
opponent, against per-opponent win rates spanning 3.3% to 49.3%. So the loop spends as much
data learning the matchups it already wins as the ones it loses.

    ogerpon_mono            5/150 =  3.3%      pairs 11.4%
    mega_abomasnow_sample  38/150 = 25.3%      pairs 11.3%
    alakazam_nz            40/150 = 26.7%      pairs 13.0%
    ...
    dragapult              74/150 = 49.3%      pairs 12.0%
    ethan_hooh             73/150 = 48.7%      pairs 12.4%

TWO GUARDS, both from things that already went wrong here.

  * A CEILING, because concentrating a round on its worst matchup is a move this project has
    already measured and lost: narrow DAgger on one deck gained +11.9pt on the target and lost
    2.75pt across the fleet ([[narrow-dagger-overfits]]). No opponent may exceed `--max-mult`
    of an even share.
  * A FLOOR, because an opponent that stops appearing stops being learned at all, and coverage
    is half of what this data does ([[winner-only-is-coverage-and-filter]]). No opponent may
    drop below `--min-mult`.

A 3.3% matchup is also a warning, not just a weight: it may be the DECK rather than the pilot,
in which case data poured into it is spent on a game that cannot be won
([[weak-decks-pilot-vs-structural]]). The floor/ceiling keep that possibility from eating a
whole round while it is still unresolved, and --report prints it so it stays visible.
"""
import argparse
import json
import sys


def allocate(rates, total, power=1.0, min_mult=0.5, max_mult=2.0):
    """{opp: win_rate 0-1} -> {opp: games}, summing to `total`.

    Weight is (1 - win_rate) ** power: a matchup we lose twice as often gets (before clamping)
    proportionally more of the budget. power is the knob the loop's ladder turns -- 0 reproduces
    the current even split exactly, which makes "did the allocation do anything" answerable by
    running the same round twice.
    """
    if not rates:
        return {}
    n = len(rates)
    even = total / float(n)
    lo, hi = min_mult * even, max_mult * even
    w = {k: max(1e-6, (1.0 - v)) ** power for k, v in rates.items()}
    s = sum(w.values())
    raw = {k: total * v / s for k, v in w.items()}

    # Clamp, then redistribute what the clamp took/gave among the still-free decks, so the
    # total is conserved exactly rather than approximately.
    out, free = {}, []
    for k, v in raw.items():
        if v < lo:
            out[k] = lo
        elif v > hi:
            out[k] = hi
        else:
            free.append(k)
    left = total - sum(out.values())
    if free:
        fs = sum(raw[k] for k in free) or 1.0
        for k in free:
            out[k] = max(lo, min(hi, left * raw[k] / fs))
    # integerise, putting the rounding remainder on the WEAKEST matchup
    ints = {k: int(v) for k, v in out.items()}
    rem = total - sum(ints.values())
    for k in sorted(rates, key=lambda x: rates[x])[:max(0, rem)]:
        ints[k] += 1
    return ints


def rates_from_gate(path, arm="cur"):
    d = json.load(open(path))
    out = {}
    for k, v in (d.get("cells") or {}).items():
        label, _, opp = k.partition("|")
        if label == arm and v.get("games"):
            out[opp] = v["win"] / float(v["games"])
    if not out:
        raise SystemExit("no cells for arm %r in %s" % (arm, path))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gate", required=True, help="gate_rN.json from gate_protagonist")
    ap.add_argument("--arm", default="cur", help="whose weakness profile to use")
    ap.add_argument("--total", type=int, required=True, help="games to spend this round")
    ap.add_argument("--power", type=float, default=1.0,
                    help="0 = even (today's behaviour); 1 = inverse win rate; >1 sharper")
    ap.add_argument("--min-mult", type=float, default=0.5)
    ap.add_argument("--max-mult", type=float, default=2.0)
    ap.add_argument("--order", default="", help="comma list: emit in THIS order")
    ap.add_argument("--report", action="store_true", help="human-readable table on stderr")
    a = ap.parse_args()

    rates = rates_from_gate(a.gate, a.arm)
    alloc = allocate(rates, a.total, a.power, a.min_mult, a.max_mult)
    order = [d for d in a.order.split(",") if d] or sorted(rates, key=lambda x: rates[x])
    missing = [d for d in order if d not in alloc]
    if missing:
        raise SystemExit("gate has no cells for %s -- refusing to emit a partial allocation, "
                         "which would silently drop those matchups" % ", ".join(missing))

    if a.report:
        even = a.total / float(len(alloc))
        print("opponent                  win%    games  vs even", file=sys.stderr)
        for d in order:
            print("  %-22s %5.1f%%  %5d  %+5.0f%%"
                  % (d, 100 * rates[d], alloc[d], 100 * (alloc[d] / even - 1)), file=sys.stderr)
        print("  %-22s %5s  %5d" % ("TOTAL", "", sum(alloc.values())), file=sys.stderr)
        worst = min(rates, key=lambda x: rates[x])
        if rates[worst] < 0.10:
            print("  NOTE: %s is at %.1f%% -- check whether that is the pilot or the DECK "
                  "before betting rounds on it" % (worst, 100 * rates[worst]), file=sys.stderr)

    print(",".join("%s=%d" % (d, alloc[d]) for d in order))


if __name__ == "__main__":
    main()
