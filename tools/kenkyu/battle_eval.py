#!/usr/bin/env python3
"""Play a trained model against the heuristic engine and record the win rate.

This is the研究's measuring instrument: every model gets the same opponent, the same deck and
alternating first player, and every run is appended to one results file so models can be
compared afterwards rather than remembered.

    # baseline 1: the heuristic engine piloting the same deck, no model at all
    python tools/kenkyu/battle_eval.py --model engine --games 100 --tag heuristic

    # baseline 2: the published model, before any of your training
    python tools/kenkyu/battle_eval.py --model base --games 40 --tag base

    # what you trained on Colab
    python tools/kenkyu/battle_eval.py --model "/content/drive/MyDrive/PTCG/models/r1" \
        --games 40 --tag r1

    # the table
    python tools/kenkyu/battle_eval.py --compare

Reading the numbers: 40 games is +-15pt of noise, 100 games about +-10pt, 400 games about
+-5pt. The table prints a 95% interval for exactly this reason -- two models whose intervals
overlap have not been told apart, however different the percentages look.
"""
import argparse
import json
import math
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT  # noqa: E402

# One results file, so every run ever made is comparable. Point it at Drive (or set
# PTCG_RESULTS) and the table is the same whether the games were played on Colab or at home.
RESULTS = os.environ.get("PTCG_RESULTS") or os.path.join(ROOT, "evaluations",
                                                         "kenkyu_results.json")
BASE_REPO = "yoheikobashi/ptcg-dusknoir-deberta-reranker"


def wilson(w, n, z=1.96):
    """95% interval for a win rate. The normal approximation is wrong at these sample sizes
    (it can reach past 100% at 40 games); Wilson's stays inside [0, 1] and is what the研究
    should quote."""
    if not n:
        return 0.0, 0.0
    p = w / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    hw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * max(0.0, c - hw), 100 * min(1.0, c + hw)


def two_prop_z(w1, n1, w2, n2):
    """z and p for "these two win rates are the same". Two-sided, normal approximation --
    good enough to say whether a gap of a few points is worth believing."""
    if not n1 or not n2:
        return 0.0, 1.0
    p1, p2 = w1 / n1, w2 / n2
    p = (w1 + w2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (p1 - p2) / se
    p_val = math.erfc(abs(z) / math.sqrt(2))
    return z, p_val


def load_results():
    if os.path.exists(RESULTS):
        with open(RESULTS, encoding="utf-8") as f:
            return json.load(f)
    return []


def save_result(rec):
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    rows = load_results()
    rows.append(rec)
    with open(RESULTS, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    return RESULTS


def compare(baseline=None, opp=None):
    rows = load_results()
    if not rows:
        print("no results yet -- run an evaluation first")
        return
    pool = {}
    for r in rows:
        if opp and r["opp"] != opp:
            continue
        k = (r["tag"], r["opp"])
        p = pool.setdefault(k, {"wins": 0, "games": 0, "runs": 0, "model": r["model"],
                                "last": r["when"]})
        p["wins"] += r["wins"]
        p["games"] += r["games"]
        p["runs"] += 1
        p["last"] = max(p["last"], r["when"])
    order = sorted(pool.items(), key=lambda kv: -kv[1]["wins"] / max(1, kv[1]["games"]))
    print("=== モデル性能比較 (同じ相手・同じデッキ・先攻後攻を交互) ===")
    print("%-16s %-18s %6s %6s %8s %-16s %s"
          % ("tag", "opponent", "games", "wins", "win%", "95% CI", "runs"))
    for (tag, o), p in order:
        lo, hi = wilson(p["wins"], p["games"])
        print("%-16s %-18s %6d %6d %7.1f%% %-16s %d"
              % (tag, o, p["games"], p["wins"], 100 * p["wins"] / p["games"],
                 "%.1f - %.1f" % (lo, hi), p["runs"]))
    base_key = None
    if baseline:
        base_key = next((k for k in pool if k[0] == baseline), None)
        if base_key is None:
            print("\n(baseline %r not in the results)" % baseline)
    if base_key is None:
        base_key = min(pool, key=lambda k: pool[k]["last"])      # the earliest run
    b = pool[base_key]
    print("\n--- %s (%d games, %.1f%%) との差 ---"
          % (base_key[0], b["games"], 100 * b["wins"] / b["games"]))
    for (tag, o), p in order:
        if (tag, o) == base_key:
            continue
        z, pv = two_prop_z(p["wins"], p["games"], b["wins"], b["games"])
        d = 100 * (p["wins"] / p["games"] - b["wins"] / b["games"])
        verdict = "有意差あり" if pv < 0.05 else "差は誤差の範囲"
        print("  %-16s %+6.1fpt   p=%.3f   %s" % (tag, d, pv, verdict))


def build_agent(spec, device, wrap, max_len, deck_name):
    """spec: 'engine' | 'base' | a checkpoint dir | a HuggingFace repo id.

    ``deck_name`` must be the deck actually dealt to this seat: make_pilot builds engine_v2
    from that deck's card list and tuning profile, and a pilot built for one deck while another
    is dealt is a silent mismatch that still plays legal moves."""
    from lm.dusk_pilot import DECK_NAME, make_pilot
    if deck_name != DECK_NAME and wrap:
        # The plan rules name dragapult_dusknoir's cards; they cannot fire for another deck.
        print("[note] plan wrapper is off: it is written for %s, not %s" % (DECK_NAME, deck_name))
        wrap = False
    if spec == "engine":
        return make_pilot(model=None, deck_name=deck_name), None
    from lm.hf_scorer import HfRerankerScorer, resolve_model
    path = resolve_model(BASE_REPO if spec == "base" else spec)
    model = HfRerankerScorer(path, device=device, max_len=max_len)
    return make_pilot(model=model, deck_name=deck_name, wrap=wrap), model


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="base", help="'engine' (no model), 'base' (the published "
                    "one), a local checkpoint dir, or a HuggingFace repo id")
    ap.add_argument("--tag", default="", help="name for this model in the comparison table "
                    "(default: derived from --model)")
    ap.add_argument("--opp", default="ogerpon_mono", help="opponent deck (its heuristic agent "
                    "pilots it)")
    ap.add_argument("--deck", default="dragapult_dusknoir", help="the deck OUR model pilots")
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--no-wrap", dest="wrap", action="store_false",
                    help="drop the hand-authored plan rules (measures a DIFFERENT pilot)")
    ap.add_argument("--compare", action="store_true", help="print the table and exit")
    ap.add_argument("--baseline", default="", help="--compare: tag to measure the others against")
    ap.add_argument("--quiet", action="store_true", help="no per-game line")
    ap.add_argument("--results", default="", help="where to keep the comparison table "
                    "(default: $PTCG_RESULTS, else evaluations/kenkyu_results.json)")
    a = ap.parse_args()
    a.model = os.path.expanduser(a.model)
    if a.results:
        global RESULTS
        RESULTS = os.path.abspath(os.path.expanduser(a.results))

    if a.compare:
        compare(a.baseline or None, a.opp)
        return

    os.chdir(ROOT)                       # deck/agent paths are relative to the repo
    import arena
    import library
    from battle_log import load_agent

    tag = a.tag or ("heuristic" if a.model == "engine"
                    else "base" if a.model == "base" else os.path.basename(a.model.rstrip("/")))
    agent, model = build_agent(a.model, a.device, a.wrap, a.max_len, a.deck)
    mine = library.read_deck(a.deck)
    theirs = library.read_deck(a.opp)
    opp_agent = load_agent(a.opp)

    print("=== %s (%s) vs %s (heuristic) : %d games ==="
          % (tag, a.model, a.opp, a.games), flush=True)
    w = n = 0
    t0 = time.time()
    secs, decs = [], []
    for g in range(a.games):
        if model is not None:
            model.reset_bank()
        seat = g % 2                     # alternate who goes first: it is worth several points
        r = (arena.play(agent, opp_agent, mine, theirs) if seat == 0
             else arena.play(opp_agent, agent, theirs, mine))
        if r is None:
            continue                     # draw / step limit: counts as neither
        n += 1
        w += (r == seat)
        if model is not None:
            secs.append(model.spent)
            decs.append(model.n_decisions)
        if not a.quiet:
            print("  game %3d/%d  %s  running %d/%d = %.1f%%"
                  % (g + 1, a.games, "WIN " if r == seat else "lose", w, n, 100 * w / n),
                  flush=True)
    lo, hi = wilson(w, n)
    el = time.time() - t0
    print("RESULT %s: %d/%d = %.1f%%  (95%% CI %.1f - %.1f)  %.0fs total, %.1fs/game"
          % (tag, w, n, 100 * w / max(1, n), lo, hi, el, el / max(1, a.games)))
    rec = {"tag": tag, "model": a.model, "deck": a.deck, "opp": a.opp, "games": n, "wins": w,
           "win_rate": 100.0 * w / max(1, n), "ci95": [lo, hi], "wrap": a.wrap,
           "device": getattr(model, "device", "engine"),
           "when": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "seconds": round(el, 1)}
    if secs:
        rec["model_seconds_per_game"] = round(sum(secs) / len(secs), 1)
        rec["decisions_per_game"] = round(sum(decs) / len(decs), 1)
        print("       model time %.1fs/game over %.1f decisions (%.2fs per decision)"
              % (rec["model_seconds_per_game"], rec["decisions_per_game"],
                 sum(secs) / max(1, sum(decs))))
    print("saved -> %s" % save_result(rec))


if __name__ == "__main__":
    main()
