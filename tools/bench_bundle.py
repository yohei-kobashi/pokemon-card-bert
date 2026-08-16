#!/usr/bin/env python3
"""Play the STAGED submission tree and measure the time bank it actually burns.

WHY NOT bench_rerank_onnx.py. That tool times the scorer over records from a rerank/pairs file
and multiplies by a decisions-per-game constant. Both inputs are wrong in the same direction
for a DPO-trained model: a pairs file holds exactly TWO candidates per decision because a pair
is the top-2 by margin, while real menus average ~5.9 and reach 24, and the cost is per
(state, candidate) pair. Benching the dusk champion that way projected 37 s/game off 2.00
candidates -- an estimate that cannot be off by a little, only by the candidate ratio.

WHAT THIS MEASURES INSTEAD. Real games, real menus, through `bundle:<stage>` -- the same main.py
the grader execs, with its baked prompt format, thread count, wrapper and embedded lm/. The
scorer's own `spent`/`n_decisions` counters are read per game, so the number reported is the
quantity Kaggle enforces rather than a projection of it.

Two things this does NOT establish. It is not a strength measurement: the win rate is printed
only as a signal that the pilot is playing rather than timing out, and the opponent field is
whatever --opp names. And it is not a Kaggle-hardware measurement -- pin it with `taskset -c
0-3` on a box whose cores are faster than Kaggle's and read the result as a lower bound.

    PYTHONPATH=cg-lib taskset -c 0-3 python tools/bench_bundle.py \
        --stage /root/subm/dusk_v1 --deck dragapult_dusknoir \
        --opp marnie_grimmsnarl,alakazam_nz --games 8
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "cg-lib"), os.path.join(ROOT, "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# The first observation of a Kaggle episode, verbatim from the competition's sample submission.
# It is also what refills the scorer's bank: mirror_env never sends it (play() returns the deck
# itself rather than asking the agent), so without this call the bank accumulates ACROSS games
# and every game after the budget is exhausted silently measures engine_v2.
_EPISODE_START = {"current": None, "logs": [], "remainingOverageTime": 600.0,
                  "search_begin_input": None, "select": None, "step": 1}


def _pct(xs, q):
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * (len(s) - 1)))]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, help="staged bundle dir (holds main.py)")
    ap.add_argument("--deck", required=True)
    ap.add_argument("--opp", required=True, help="comma-separated opponent decks")
    ap.add_argument("--games", type=int, default=8, help="per opponent; seats alternate")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--budget", type=float, default=600.0,
                    help="the bank the RULES enforce, for the headroom column. The scorer's own "
                         "budget is baked into the bundle and is deliberately lower.")
    ap.add_argument("--mirror-so", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    import mirror_match as mm
    from tools.mirror_env import DEFAULT_SO, MirrorEngine, play

    eng = MirrorEngine(a.mirror_so or DEFAULT_SO)
    tuning = json.load(open(os.path.join(ROOT, "agents", "tuning.json")))
    my_ids = mm.load_deck(a.deck)

    # `bundle:` execs the staged main.py and REFUSES a tree that loaded below tier reranker, so
    # a bundle that silently degraded to engine_v2 cannot be benched as if it were the model.
    agent, sc = mm.make_agent("bundle:" + a.stage, a.deck, my_ids, tuning.get(a.deck, {}))
    if sc is None:
        raise SystemExit("the bundle exposed no _scorer -- nothing to measure")

    # THE DEADLOCK COUNTER. Live games are lost to a board state the win rate cannot see: our
    # active holds no energy, so it cannot pay its retreat cost, so the loaded attacker on the
    # bench can never be promoted and we never attack again. Measured on the six live replays of
    # submission 55445834 it held on 10 of our turns, and on all 10 neither retreat nor attack
    # was offered. It barely reproduces against engine_v2, which is exactly why a local gate that
    # reports only a win rate cleared a pilot that then rated ~330 live. Counted here by watching
    # the observations the agent is handed, so it costs nothing and cannot be forgotten.
    seen_turn = set()
    stat = {"turns": 0, "active_dry": 0, "deadlock": 0}

    def _watch(obs):
        cur = obs.get("current") or {}
        pl = cur.get("players") or []
        yi = cur.get("yourIndex")
        if not pl or yi is None or yi >= len(pl):
            return
        key = (len(rows), cur.get("turn"))
        if key in seen_turn:
            return
        seen_turn.add(key)
        p = pl[yi]
        act = p.get("active")
        if isinstance(act, list):
            act = act[0] if act else None
        if not isinstance(act, dict):
            return
        stat["turns"] += 1
        if len(act.get("energies") or []) == 0:
            stat["active_dry"] += 1
            if any(len(b.get("energies") or []) >= 2 for b in (p.get("bench") or [])):
                stat["deadlock"] += 1

    inner = agent

    def agent(obs):                                        # noqa: F811 -- deliberate rebind
        try:
            _watch(obs)
        except Exception:                                  # noqa: BLE001
            pass                                           # a diagnostic must never pilot
        return inner(obs)

    rows = []
    wins = 0
    n = 0
    for opp in [o for o in a.opp.split(",") if o]:
        opp_ids = mm.load_deck(opp)
        opp_agent, _ = mm.make_agent("engine", opp, opp_ids, tuning.get(opp, {}))
        for g in range(a.games):
            agent(_EPISODE_START)             # refills the bank; see _EPISODE_START
            before_n = int(getattr(sc, "n_decisions", 0) or 0)
            t0 = time.time()
            seed = a.seed + g // 2
            mine = g % 2
            r = (play(eng, agent, opp_agent, my_ids, opp_ids, seed, mirror=1) if mine == 0
                 else play(eng, opp_agent, agent, opp_ids, my_ids, seed, mirror=1))
            spent = float(getattr(sc, "spent", 0.0) or 0.0)
            dec = int(getattr(sc, "n_decisions", 0) or 0) - before_n
            rows.append({"opp": opp, "seed": seed, "seat": mine, "spent_s": round(spent, 2),
                         "decisions": dec, "wall_s": round(time.time() - t0, 1),
                         "won": int(r == mine)})
            wins += int(r == mine)
            n += 1
            print("  %-22s seed %-6d seat %d | bank %6.1fs over %3d decisions | %s"
                  % (opp, seed, mine, spent, dec, "W" if r == mine else "L"), flush=True)

    banks = [x["spent_s"] for x in rows]
    decs = [x["decisions"] for x in rows]

    # A bank that only ever grows is not a slow pilot -- it is a pilot whose bank never resets,
    # and every number below it is then cumulative rather than per-game. That is a real bundle
    # bug (main.py returning the deck on select=None without forwarding the call, so lm/agent's
    # reset_bank is unreachable), and it is invisible in the totals: the first game looks fine.
    # Checked here because this is the only place that plays a bundle for more than one game.
    if len(banks) > 2 and all(b <= c for b, c in zip(banks, banks[1:])):
        print("\n!! the bank NEVER decreased across %d games -- reset_bank is not wired up.\n"
              "!! Every figure below is cumulative, not per-game; fix the bundle and re-run."
              % len(banks), flush=True)
    # A game that hits the scorer's own budget does not fail -- the scorer raises and lm/agent
    # falls back to engine_v2 for the rest of the game. It is not a crash, it is a SILENT
    # DOWNGRADE, and it is the thing this bench exists to catch before the ladder does.
    hit = sum(1 for b in banks if b >= float(getattr(sc, "time_budget", 1e9)) - 1.0)
    print("\n%d games vs %s" % (n, a.opp))
    print("  bank/game       mean %6.1fs  p90 %6.1fs  max %6.1fs  (scorer budget %.0fs, "
          "rules %.0fs)" % (sum(banks) / max(1, n), _pct(banks, 0.9), max(banks or [0]),
                            float(getattr(sc, "time_budget", 0.0) or 0.0), a.budget))
    print("  decisions/game  mean %6.1f   p90 %6.0f   max %6.0f" %
          (sum(decs) / max(1, n), _pct(decs, 0.9), max(decs or [0])))
    print("  worst game uses %.1f%% of the %.0fs bank" % (100.0 * max(banks or [0]) / a.budget,
                                                          a.budget))
    print("  games that exhausted the scorer budget: %d/%d%s"
          % (hit, n, "  <-- those finished as engine_v2" if hit else ""))
    print("  win rate %d/%d = %.1f%% (a liveness signal, NOT a strength measurement)"
          % (wins, n, 100.0 * wins / max(1, n)))
    tt = max(1, stat["turns"])
    print("  BOARD: active with no energy %d/%d turns (%.0f%%); of those, a bench Pokemon held "
          ">=2 energy on %d (%.0f%% of all turns) <- the live loss mode"
          % (stat["active_dry"], stat["turns"], 100.0 * stat["active_dry"] / tt,
             stat["deadlock"], 100.0 * stat["deadlock"] / tt))

    if a.out:
        with open(a.out, "w") as f:
            json.dump({"stage": a.stage, "deck": a.deck, "games": n, "rows": rows,
                       "bank_mean_s": sum(banks) / max(1, n), "bank_max_s": max(banks or [0]),
                       "decisions_max": max(decs or [0]), "budget_hits": hit,
                       "board": stat}, f, indent=1)
        print("-> %s" % a.out)


if __name__ == "__main__":
    main()
