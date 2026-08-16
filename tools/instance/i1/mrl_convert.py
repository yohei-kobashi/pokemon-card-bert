#!/usr/bin/env python3
"""Branch pairs -> listwise training rows for the mirror-RL loop.

    wc = (1 - BETA) * softmax([qw, ql] / T)  +  BETA * rule_share

qw/ql already carry the whole outcome term: win/loss +-1 plus the prize margin
(rl_branch.PRIZE_GAMMA, set by the chain). BETA keeps the rule term subordinate on purpose --
conformance alone made the pilot 3.6x worse (plan-conformance-is-not-winning), so rules act as
a tiebreaker on decisions the playouts can barely separate, not as a second objective.

Rows where the two candidates rendered to the same text are dropped (nothing to rank), as are
rows whose Q gap and rule gap are both zero (no gradient either way).
"""
import argparse
import gzip
import json
import math

ap = argparse.ArgumentParser()
ap.add_argument("--pairs", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--beta", type=float, default=0.3)
ap.add_argument("--temp", type=float, default=0.5)
ap.add_argument("--phi-min", type=float, default=0.0,
                help="rescue below-qmin pairs whose setup potentials differ by at least this. "
                     "Phi steps in halves (one useful energy = 0.5). 0 disables.")
ap.add_argument("--phi-wc", type=float, default=0.65,
                help="the winner's weight on a phi-labelled row. Deliberately weak: this is a "
                     "prior about setup, not a measured outcome, and forcing the same "
                     "preference at inference cost -2.25pt.")
ap.add_argument("--qmin", type=float, default=0.0,
                help="drop pairs whose playout advantage |qw-ql| is below this. "
                     "The Q estimate from 24 playouts has an SE around 0.2 and the "
                     "median pair margin is 0.26, so most pairs are coin flips -- and "
                     "measured, they do not merely add nothing: training on all of them "
                     "moved held-out conformance 54.3 -> 53.6, while >=0.35 moved it "
                     "52.1 -> 58.1. The low-margin majority outvotes the signal.")
a = ap.parse_args()

n_in = n_out = n_same = n_flat = n_weak = n_phi = 0
with gzip.open(a.out, "wt") as out:
    for line in gzip.open(a.pairs, "rt"):
        d = json.loads(line)
        n_in += 1
        cw, cl = d.get("cw"), d.get("cl")
        if not cw or not cl or cw == cl:
            n_same += 1
            continue
        qw, ql = float(d["qw"]), float(d["ql"])
        rw, rl = float(d.get("rww") or 0.0), float(d.get("rwl") or 0.0)
        if abs(qw - ql) < 1e-9 and abs(rw - rl) < 1e-9:
            n_flat += 1
            continue
        if abs(qw - ql) < a.qmin:
            pw, pl_ = d.get("phi_w"), d.get("phi_l")
            if (a.phi_min > 0 and pw is not None and pl_ is not None
                    and abs(pw - pl_) >= a.phi_min):
                # Q could not separate these; the board can. Order by the potential and write a
                # WEAK label rather than dropping the row -- 64% of pairs land here.
                hi, lo = ((cw, cl) if pw > pl_ else (cl, cw))
                out.write(json.dumps({"prompt": d["prompt"], "cands": [hi, lo],
                                      "wc": [round(a.phi_wc, 4), round(1 - a.phi_wc, 4)]}) + "\n")
                n_phi += 1
                n_out += 1
                continue
            n_weak += 1
            continue
        ew, el = math.exp(qw / a.temp), math.exp(ql / a.temp)
        soft = (ew / (ew + el), el / (ew + el))
        rs = rw + rl
        rshare = (rw / rs, rl / rs) if rs > 0 else (0.5, 0.5)
        wc = [(1 - a.beta) * soft[i] + a.beta * rshare[i] for i in (0, 1)]
        out.write(json.dumps({"prompt": d["prompt"], "cands": [cw, cl],
                              "wc": [round(x, 4) for x in wc]}) + "\n")
        n_out += 1
print("[mrl] %d pairs -> %d rows (same-text %d, flat %d, below-qmin %d, phi-labelled %d)"
      " | beta %.2f temp %.2f qmin %.2f phi-min %.2f"
      % (n_in, n_out, n_same, n_flat, n_weak, n_phi, a.beta, a.temp, a.qmin, a.phi_min))
