"""P4' acceptance (pipeline v2): BUNDLE-level A/B of engine_v2 vs the legacy
agent over the standard field panel, with an adaptive sample size and a 95% CI.

Protocol (docs/l2_pipeline.md v2):
  - within-run v2-vs-legacy Delta only (legacy WR swings +-10pt across runs)
  - phase 1: 48 games/opponent (~n=370); if |Delta| < 2*SE, run phase 2 (double)
  - report Delta with 95% CI; acceptance = CI upper bound >= 0 (parity or better)
  - the unit under test is a LINE BUNDLE, never a single rule (combo edges are
    individually-negative by design; greedy per-rule A/B would reject them)

Usage: PYTHONPATH=cg-lib python tools/p4_accept.py <deck>[,<deck>...]
"""
import os, sys, json, math
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "cg-lib"), os.path.join(ROOT, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)
import arena, library
from multiprocessing import Pool
from battle_log import load_agent

TUN = json.load(open(os.path.join(ROOT, "agents", "tuning.json")))
PANEL = ["zangoose", "mega_venusaur", "dragapult", "crustle_stall", "archaludon",
         "hop_zacian", "mega_gardevoir", "slowking"]
BATCH = 6           # games per isolated worker (RNG decorrelation)
PER_OPP = 48        # phase-1 games per opponent per side


def task(a):
    deck, which, panel, batch = a
    from agents import engine_v2
    dk = library.read_deck(deck)
    dp = library.read_deck(panel)
    me = (engine_v2.make_policy(dk, TUN[deck]).act if which == "v2" else load_agent(deck))
    opp = engine_v2.make_policy(dp, TUN.get(panel, {})).act
    wa, wb = arena.match(me, dk, opp, dp, games=batch)
    return (deck, which, wa, wa + wb)


def run_phase(deck, per_opp):
    tasks = []
    for pn in [x for x in PANEL if x != deck]:
        for which in ("v2", "legacy"):
            n = 0
            while n < per_opp:
                b = min(BATCH, per_opp - n)
                tasks.append((deck, which, pn, b))
                n += b
    with Pool(20, maxtasksperchild=1) as pool:
        rows = pool.map(task, tasks)
    agg = {"v2": [0, 0], "legacy": [0, 0]}
    for _, which, w, g in rows:
        agg[which][0] += w
        agg[which][1] += g
    return agg


def report(deck, agg, phase):
    (w1, g1), (w0, g0) = agg["v2"], agg["legacy"]
    p1, p0 = w1 / g1, w0 / g0
    d = p1 - p0
    se = math.sqrt(p1 * (1 - p1) / g1 + p0 * (1 - p0) / g0)
    lo, hi = d - 1.96 * se, d + 1.96 * se
    verdict = "ACCEPT (>= legacy)" if lo > 0 else (
        "ACCEPT (parity: CI covers 0)" if hi >= 0 else "REJECT (< legacy)")
    print(f"{deck:20} phase{phase} v2 {100*p1:.1f}% legacy {100*p0:.1f}% "
          f"Δ{100*d:+.1f} CI[{100*lo:+.1f},{100*hi:+.1f}] (n={g1}) -> {verdict}")
    return d, se


def main():
    for deck in sys.argv[1].split(","):
        agg = run_phase(deck, PER_OPP)
        d, se = report(deck, agg, 1)
        if abs(d) < 2 * se:                      # noise-ambiguous: extend, don't declare
            more = run_phase(deck, PER_OPP)
            for k in agg:
                agg[k][0] += more[k][0]
                agg[k][1] += more[k][1]
            report(deck, agg, 2)


if __name__ == "__main__":
    main()
