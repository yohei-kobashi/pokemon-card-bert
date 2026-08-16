#!/usr/bin/env python3
"""Per-rule conformance on LIVE games: when a plan rule fired, did we actually obey it?

Answers a question the win rate cannot: the deck may be losing because it is outclassed, or
because it is not doing the things we told it to do, and those call for opposite work.

Two things make this honest:

* **CHANCE BASELINE.** Raw conformance is meaningless on its own -- a rule whose conformant set
  is 4 of 5 options is "obeyed" 80% of the time by a coin. Every row reports the mean chance of
  conformance (|good| / |options|) over the menus where the rule fired, so the column that
  matters is the LIFT over it.
* **WHO OWNS THE DECISION.** The five R5 rules are FILTERED by the shipped planfilter wrapper --
  the menu handed to the model is pre-restricted to conformant options, so conformance near 100%
  is the wrapper working, not the model agreeing. The other eleven are the model's own choices
  and are the only place a conformance number says something about the model.
* **PER-MENU vs PER-TURN, and they answer different questions.** A rule like "attack with Phantom
  Dive when it is legal" fires on EVERY menu of the turn in which the attack is legal, and only
  one of them can be the attack -- the rest are the draws, attaches and evolutions that correctly
  come first. Scoring it per menu reads 39.1%; scoring it per turn reads 94.7%, and the second is
  the one that means "did we do what the plan said". I reported the 39.1% as the deck's largest
  piloting defect before noticing this. Prohibitions ("do not Judge while...") are the opposite:
  they must hold at EVERY menu, so per-menu is their real number and per-turn would flatter them.
  Both columns are printed; read the one that matches the rule's shape.

Requires replays exported with `tools/export_live_logs.py --raw`, which keeps the real `obs`.
Reconstructing an obs from the trimmed export does NOT work: on the trimmed files only
phantom_dive and boss_damaged ever fire, which reads as "the rules never engage" and is an
artifact of the missing fields.

    PYTHONPATH=cg-lib:tools python tools/live_rule_audit.py --logs /root/livelogs_raw
"""
import argparse
import collections
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "cg-lib"), os.path.join(ROOT, "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

R5 = ("lethal_now", "spread_aim", "clops_hold", "energy_line", "energy_focus")
OURS = "Yohei-Kobashi"


def our_seat(path):
    """Board index of our agent, from the exporter's `{label0}_vs_{label1}` convention (labels
    are ordered by board index). Returns None for our own mirror, where the question is ill-posed."""
    stem = os.path.basename(path).rsplit(".", 1)[0]
    try:
        a, b = stem.split("_vs_", 1)
    except ValueError:
        return None
    a_ours, b_ours = OURS in a, OURS in b
    if a_ours and b_ours:
        return None
    if a_ours:
        return 0
    return 1 if b_ours else None


def opponent(path):
    stem = os.path.basename(path).rsplit(".", 1)[0]
    part = stem.split("_vs_", 1)[-1]
    return part[:30]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", required=True, help="dir of --raw exported replays")
    ap.add_argument("--by-opponent", action="store_true")
    a = ap.parse_args()

    import dusk_plan

    # rule -> [fired, obeyed, sum_chance]
    agg = collections.defaultdict(lambda: [0, 0, 0.0])
    per_opp = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0, 0.0]))
    # rule -> (file, turn) -> [fired, obeyed]; a turn counts as obeyed if ANY menu in it complied
    per_turn = collections.defaultdict(dict)
    files = skipped = decisions = 0

    for f in sorted(glob.glob(os.path.join(a.logs, "*.json"))):
        seat = our_seat(f)
        if seat is None:
            skipped += 1
            continue
        files += 1
        opp = opponent(f)
        for el in json.load(open(f)):
            o = el.get("obs")
            sel = el.get("selected")
            if not isinstance(o, dict) or not sel:
                continue
            cur = o.get("current") or {}
            if cur.get("yourIndex") != seat:
                continue                       # the opponent's decision, not ours
            opts = (o.get("select") or {}).get("option") or []
            if len(opts) < 2:
                continue
            decisions += 1
            try:
                live = dusk_plan.opportunities(o)
            except Exception:                  # noqa: BLE001
                continue
            picked = set(sel if isinstance(sel, (list, tuple)) else [sel])
            for r, hit in live.items():
                if not hit:
                    continue
                good = set(hit[0])
                if not good:
                    continue
                ok = bool(good & picked)
                chance = len(good) / float(len(opts))
                for tgt in (agg[r], per_opp[opp][r]):
                    tgt[0] += 1
                    tgt[1] += int(ok)
                    tgt[2] += chance
                tk = (f, cur.get("turn"))
                slot = per_turn[r].setdefault(tk, [0, 0])
                slot[0] = 1
                slot[1] |= int(ok)

    print("%d games (skipped %d self-mirror), %d of OUR multi-option decisions\n"
          % (files, skipped, decisions))
    print("%-16s %6s %7s %8s %7s %7s %8s   owner" %
          ("rule", "menus", "obey%", "chance%", "lift", "turns", "perTurn%"))
    rows = sorted(agg.items(), key=lambda kv: -kv[1][0])
    for r, (n, ok, ch) in rows:
        obey = 100.0 * ok / n
        chance = 100.0 * ch / n
        tk = per_turn.get(r, {})
        nt = len(tk)
        okt = sum(v[1] for v in tk.values())
        print("%-16s %6d %6.1f%% %7.1f%% %+6.1f %7d %7.1f%%   %s"
              % (r, n, obey, chance, obey - chance, nt,
                 100.0 * okt / nt if nt else 0.0,
                 "WRAPPER (pre-filtered)" if r in R5 else "model"))

    model_rows = [(r, v) for r, v in rows if r not in R5]
    if model_rows:
        n = sum(v[0] for _r, v in model_rows)
        ok = sum(v[1] for _r, v in model_rows)
        ch = sum(v[2] for _r, v in model_rows)
        print("\nMODEL-OWNED rules overall: %.1f%% obeyed vs %.1f%% chance (lift %+.1f) over %d firings"
              % (100.0 * ok / n, 100.0 * ch / n, 100.0 * (ok - ch) / n, n))

    if a.by_opponent:
        print("\n--- by opponent (model-owned rules only) ---")
        for opp, d in sorted(per_opp.items()):
            n = sum(v[0] for r, v in d.items() if r not in R5)
            ok = sum(v[1] for r, v in d.items() if r not in R5)
            ch = sum(v[2] for r, v in d.items() if r not in R5)
            if not n:
                continue
            print("  %-32s %5.1f%% vs %5.1f%% chance (lift %+5.1f, n=%d)"
                  % (opp, 100.0 * ok / n, 100.0 * ch / n, 100.0 * (ok - ch) / n, n))


if __name__ == "__main__":
    main()
