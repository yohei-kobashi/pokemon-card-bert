#!/usr/bin/env python3
"""One deck against a WEIGHTED FIELD of opponents, engine_v2 on both sides.

The screens we already had answer a different question. `mirror_match --mirror` puts the same
decklist on both seats, which measures a PILOT against another pilot and cancels the matchup;
the fleet round-robin measures every deck against every deck, which spends 62/63 of its games
on opponents nobody plays. Neither tells you what a deck scores against the field it will
actually meet, and that is the number that decides whether a rebuild worked.

So: play `--deck` against each opponent in `--field`, seats alternating, and report both the
per-matchup rate and the rate weighted by the opponents' real share of the ladder. A rebuild
that gains 10pt against a 2%-of-the-field deck and loses 3pt to a 30% one is a regression, and
only the weighted number says so.

Determinism: seeds are fixed per (matchup, game), so two runs of the same decks compare
pairwise. Different decklists legitimately give different shuffles from the same seed -- the
seed fixes the ENGINE's stream, not the cards -- so a before/after on a REBUILT deck is not
paired and needs the games to carry it.

    PYTHONPATH=cg-lib python3 tools/eval_deck_field.py --deck slowking \\
        --field marnie_grimmsnarl:26,dudunsparce_box:18,alakazam_nz:12 --games 60
"""

import argparse
import collections
import json
import multiprocessing as mp
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "cg-lib"), os.path.join(ROOT, "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_ENG = None
_AG = {}


def _agent(deck):
    """engine_v2 for `deck`, built once per worker. Building it per game re-reads tuning.json
    and re-resolves card_roles for every battle, which dominates the run at 60 games."""
    global _AG
    if deck not in _AG:
        import library
        from lm.agent import make_lm_agent
        ids = [int(x) for x in open(library.deck_path(deck)) if x.strip()]
        tuning = json.load(open(os.path.join(ROOT, "agents", "tuning.json")))
        _AG[deck] = (ids, make_lm_agent(ids, tuning.get(deck, {}), model=None))
    return _AG[deck]


def _one(job):
    deck, opp, seed, flip, so, max_steps = job
    global _ENG
    from mirror_env import MirrorEngine, play
    # Engine only inside the worker, never in the parent: a native library initialised before
    # fork() leaves the children sharing one engine's state and dies as a C++ terminate.
    if _ENG is None:
        _ENG = MirrorEngine(so)
    ids_me, ag_me = _agent(deck)
    ids_op, ag_op = _agent(opp)
    try:
        if flip:
            r = play(_ENG, ag_op, ag_me, ids_op, ids_me, seed, mirror=1, max_steps=max_steps)
            won = 1 if r == 1 else (0 if r == 0 else None)
        else:
            r = play(_ENG, ag_me, ag_op, ids_me, ids_op, seed, mirror=1, max_steps=max_steps)
            won = 1 if r == 0 else (0 if r == 1 else None)
    except Exception:                                          # noqa: BLE001
        return opp, None, flip
    return opp, won, flip


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deck", required=True)
    ap.add_argument("--field", required=True,
                    help="comma list of opponent[:weight]; weights default to 1 and are "
                         "used only for the weighted summary, never for how many games "
                         "each matchup gets -- a 2%% deck still needs enough games to read")
    ap.add_argument("--games", type=int, default=60, help="per opponent")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--seed", type=int, default=90000)
    ap.add_argument("--max-steps", type=int, default=4000)
    ap.add_argument("--mirror-so", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--label", default="")
    a = ap.parse_args()

    import library
    from mirror_env import DEFAULT_SO
    so = a.mirror_so or DEFAULT_SO
    known = set(library.list_decks())
    field = []
    for tok in a.field.split(","):
        if not tok.strip():
            continue
        name, _, w = tok.partition(":")
        if name not in known:
            sys.exit("unknown opponent: %s" % name)
        field.append((name, float(w) if w else 1.0))
    if a.deck not in known:
        sys.exit("unknown deck: %s" % a.deck)

    jobs = []
    for oi, (opp, _) in enumerate(field):
        for g in range(a.games):
            jobs.append((a.deck, opp, a.seed + 7919 * oi + g, g % 2, so, a.max_steps))
    print("%s vs %d opponents x %d games = %d battles on %d workers"
          % (a.deck, len(field), a.games, len(jobs), a.workers), flush=True)

    t0 = time.time()
    w = collections.Counter()
    n = collections.Counter()
    seat = collections.defaultdict(lambda: [0, 0, 0, 0])   # opp -> [w0,n0,w1,n1]
    draws = collections.Counter()
    with mp.get_context("spawn").Pool(min(a.workers, max(1, len(jobs)))) as p:
        for k, (opp, won, flip) in enumerate(p.imap_unordered(_one, jobs, chunksize=4)):
            if won is None:
                draws[opp] += 1
            else:
                w[opp] += won
                n[opp] += 1
                s = seat[opp]
                s[2 * flip + 1] += 1
                s[2 * flip] += won
            if (k + 1) % 200 == 0:
                print("  %d/%d  %.0fs" % (k + 1, len(jobs), time.time() - t0), flush=True)

    print("\n%-24s %8s %8s %8s %7s %7s" % ("opponent", "wr", "games", "draws", "seat0", "seat1"))
    tot_w = tot_n = 0.0
    wsum = wtot = 0.0
    rows = {}
    for opp, wt in field:
        if not n[opp]:
            print("%-24s %8s (no decided games)" % (opp, "-"))
            continue
        p_ = w[opp] / n[opp]
        s = seat[opp]
        rows[opp] = {"w": w[opp], "n": n[opp], "p": p_, "weight": wt,
                     "seat0": [s[0], s[1]], "seat1": [s[2], s[3]], "draws": draws[opp]}
        print("%-24s %7.1f%% %8d %8d %6.1f%% %6.1f%%"
              % (opp, 100 * p_, n[opp], draws[opp],
                 100 * s[0] / s[1] if s[1] else float("nan"),
                 100 * s[2] / s[3] if s[3] else float("nan")))
        tot_w += w[opp]
        tot_n += n[opp]
        wsum += wt * p_
        wtot += wt
    if tot_n:
        print("\nunweighted %.1f%% over %d decided games (%d draws)"
              % (100 * tot_w / tot_n, tot_n, sum(draws.values())))
    if wtot:
        print("FIELD-WEIGHTED %.1f%%   <- the number that decides" % (100 * wsum / wtot))
    if a.out:
        json.dump({"deck": a.deck, "label": a.label, "games": a.games,
                   "weighted": (wsum / wtot) if wtot else None,
                   "unweighted": (tot_w / tot_n) if tot_n else None,
                   "decks": rows}, open(a.out, "w"), indent=1)
        print("-> %s" % a.out)


if __name__ == "__main__":
    main()
