"""Score an RL gate: pool an eval_rerank_par.sh output dir, optionally MINUS the engine_v2
baseline on the same cells, and print one fraction.

Why a difference. The goal is "reach engine_v2's level", and absolute win rates cannot
express it: on the Stage-C grid engine_v2 itself averages 37.4% and gets 24.0% with dragapult
(measured 2026-07-28, 300 games/cell). An absolute threshold is simultaneously unreachable on
the hard half of the grid and free on the easy half, and moves when the deck set changes. The
delta is scale-free: 0 means the policy pilots as well as the engine, which is the target.

Pooling rules, both load-bearing:
  * games are SUMMED across cells, not averaged over cells, so a cell whose process died
    (its json simply absent) cannot silently reweight the rest;
  * with --baseline, only cells present in BOTH are counted, and the baseline is pooled over
    exactly that subset -- otherwise a missing LM cell would be compared against a baseline
    that still includes it.

    python tools/rl_gate_score.py <lm_outdir> [--baseline <dir>] [--verbose]
"""
import json
import os
import sys


def _norm(k):
    """'crustle (RR) vs alakazam (heuristic)' -> ('crustle', 'alakazam')."""
    if " vs " in k:
        left, right = k.split(" vs ", 1)
        return left.split(" (")[0].strip(), right.split(" (")[0].strip()
    return k, ""


def load_lm(d):
    out = {}
    if not os.path.isdir(d):
        return out
    for f in sorted(os.listdir(d)):
        if not f.endswith(".json"):
            continue
        for k, v in json.load(open(os.path.join(d, f))).get("results", {}).items():
            out[_norm(k)] = (v["win"], v["games"])
    return out


def load_base(d):
    out = {}
    if not d or not os.path.isdir(d):
        return out
    for f in sorted(os.listdir(d)):
        if not f.endswith(".json"):
            continue
        r = json.load(open(os.path.join(d, f)))
        if "deck" in r and "opp" in r:                       # rl_baseline_cell.py
            out[(r["deck"], r["opp"])] = (r["win"], r["games"])
        else:                                                # {"deck vs opp": pct}
            for k, v in r.items():
                key = _norm(k)
                out[key] = (v * 3.0, 300)                    # pct -> wins at 300 games
    return out


def main():
    args = sys.argv[1:]
    lm_dir = args[0]
    base_dir = ""
    if "--baseline" in args:
        base_dir = args[args.index("--baseline") + 1]
    verbose = "--verbose" in args

    lm = load_lm(lm_dir)
    base = load_base(base_dir)
    cells = [c for c in lm if not base or c in base]

    lw = sum(lm[c][0] for c in cells)
    ln = sum(lm[c][1] for c in cells)
    lm_wr = lw / ln if ln else 0.0
    if not base:
        if verbose:
            sys.stderr.write("gate: %d cells, %d games, LM %.1f%%\n"
                             % (len(cells), ln, 100 * lm_wr))
        print(round(lm_wr, 4) if ln else 0.0)
        return

    bw = sum(base[c][0] for c in cells)
    bn = sum(base[c][1] for c in cells)
    base_wr = bw / bn if bn else 0.0
    if verbose:
        sys.stderr.write("gate: %d cells | LM %.1f%% (%d games) | engine_v2 %.1f%% (%d games)"
                         " | delta %+.1fpt\n"
                         % (len(cells), 100 * lm_wr, ln, 100 * base_wr, bn,
                            100 * (lm_wr - base_wr)))
        for c in sorted(cells):
            sys.stderr.write("    %-20s vs %-18s LM %5.1f  engine %5.1f  %+6.1f\n"
                             % (c[0], c[1], 100 * lm[c][0] / lm[c][1],
                                100 * base[c][0] / base[c][1],
                                100 * (lm[c][0] / lm[c][1] - base[c][0] / base[c][1])))
    print(round(lm_wr - base_wr, 4) if ln and bn else 0.0)


if __name__ == "__main__":
    main()
