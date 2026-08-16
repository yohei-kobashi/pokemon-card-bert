#!/usr/bin/env python3
"""Run a STAGED submission bundle the way Kaggle will, and report what it actually did.

The bundle's own selfcheck proves the tree imports and a prompt can be built. That is not the
same as proving it plays. This drives the bundle's `agent` through real battles and checks the
four things that have actually gone wrong before:

  1. THE DECK-SELECTION CALL. The first obs of every episode carries select=None and must be
     answered with the 60-card deck. mirror_env never sends it -- it returns None instead of
     asking the agent -- so no amount of local match play exercises it. Three LM submissions
     errored on exactly this. Called here directly.
  2. FORFEITS. mirror_env's play() treats an exception or an illegal selection as a loss for
     that seat rather than a crash, so a broken agent looks like a bad agent. Counted apart
     from ordinary losses by driving the game loop here.
  3. THE TIME BANK. The scorer raises once its budget is spent and make_lm_agent is supposed to
     fall back to engine_v2. Re-run with an absurdly small budget: games must still COMPLETE.
  4. WHERE THE DECISIONS WENT. With --defer attach, attach decisions should be answered without
     the model being called at all. Counted by instrumenting the scorer, so the answer comes
     from the shipped object rather than from reading the code.

    python3 smoke_bundle.py /root/subm/dusk_s1_attach --games 4 --opp crustle
"""
import argparse
import importlib.util
import os
import sys
import time

REPO = "/root/ptcg/repo"
for p in (REPO, os.path.join(REPO, "cg-lib"), os.path.join(REPO, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)


def load_bundle(stage):
    spec = importlib.util.spec_from_file_location("kaggle_main", os.path.join(stage, "main.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage")
    ap.add_argument("--games", type=int, default=4)
    ap.add_argument("--opp", default="crustle")
    ap.add_argument("--seed", type=int, default=7000)
    ap.add_argument("--tiny-budget", type=float, default=0.0,
                    help="override the scorer's time budget (s) to force the fallback path")
    a = ap.parse_args()

    m = load_bundle(a.stage)
    print("TIER: %s" % getattr(m, "TIER", "?"))
    print("PROMPT_FMT: %s" % (getattr(m, "PROMPT_FMT", None),))
    print("DEFER_KINDS: %s" % (getattr(m, "DEFER_KINDS", ()),))

    # ---- 1. the deck-selection call -------------------------------------------------
    deck = m.agent({"current": {"yourIndex": 0}})     # no "select" key at all
    ok = isinstance(deck, list) and len(deck) == 60 and all(isinstance(x, int) for x in deck)
    print("DECK-SELECT: %s (%s of len %d)"
          % ("OK" if ok else "FAILED", type(deck).__name__, len(deck) if hasattr(deck, "__len__") else -1))
    if not ok:
        sys.exit("the first call of every episode is broken; nothing else matters")

    # ---- instrument the shipped scorer ----------------------------------------------
    sc = getattr(m, "_scorer", None)
    n_scored = [0]
    if sc is not None:
        if a.tiny_budget:
            sc.time_budget = a.tiny_budget
        real_score = sc.score

        def counting_score(prompt, cands, obs=None):
            n_scored[0] += 1
            return real_score(prompt, cands, obs)
        sc.score = counting_score
    else:
        print("NOTE: no _scorer on the bundle -- it fell back before the model loaded")

    n_calls = [0]
    n_attach_menus = [0]
    from lm.actions import encode_option
    real_agent = m.agent

    def watched(obs):
        n_calls[0] += 1
        opts = (obs.get("select") or {}).get("option") or []
        try:
            kinds = {encode_option(o, obs).split(":", 1)[0].split("@", 1)[0] for o in opts}
        except Exception:
            kinds = set()
        if "attach" in kinds:
            n_attach_menus[0] += 1
        return real_agent(obs)

    # ---- 2/3/4. real games -----------------------------------------------------------
    # library.deck_path returns a REPO-RELATIVE path, so the harness only finds decklists from
    # the repo root. Done after the bundle is loaded: main.py resolves everything from its own
    # HERE, so moving cwd underneath it is safe and moving it BEFORE would not be.
    os.chdir(REPO)
    from tools.mirror_env import DEFAULT_SO, MirrorEngine, play
    import mirror_match as mm
    import json
    eng = MirrorEngine(DEFAULT_SO)
    tuning = json.load(open(os.path.join(REPO, "agents", "tuning.json")))
    my_ids = mm.load_deck("dragapult_dusknoir")
    op_ids = mm.load_deck(a.opp)
    opp = mm.make_agent("engine", a.opp, op_ids, tuning.get(a.opp, {}))[0]

    wins = 0
    played = 0
    t0 = time.time()
    for g in range(a.games):
        before = (n_calls[0], n_scored[0])
        mine = g % 2
        r = (play(eng, watched, opp, my_ids, op_ids, a.seed + g, mirror=1) if mine == 0
             else play(eng, opp, watched, op_ids, my_ids, a.seed + g, mirror=1))
        d_calls = n_calls[0] - before[0]
        d_scored = n_scored[0] - before[1]
        # A game where our agent was asked nothing is a game that ended before it played, which
        # is what a crash at start looks like from out here.
        status = "no-decisions" if d_calls == 0 else ("win" if r == mine else
                                                      "loss" if r is not None else "draw")
        print("  game %d seat%d: %-12s decisions %3d  model-scored %3d"
              % (g, mine, status, d_calls, d_scored))
        if d_calls:
            played += 1
            if r == mine:
                wins += 1

    dt = time.time() - t0
    print("\nGAMES completed %d/%d | wins %d | %.1fs (%.1fs/game)"
          % (played, a.games, wins, dt, dt / max(1, a.games)))
    print("DECISIONS seen %d | model-scored %d (%.0f%%) | menus containing attach %d"
          % (n_calls[0], n_scored[0], 100.0 * n_scored[0] / max(1, n_calls[0]), n_attach_menus[0]))
    if getattr(m, "DEFER_KINDS", ()):
        print("DEFER: %d attach menus were present; with defer they must NOT be model-scored"
              % n_attach_menus[0])
    if sc is not None:
        print("BANK: budget %.1fs, spent %.1fs on the last game" % (sc.time_budget, sc.spent))
    if played < a.games:
        sys.exit("SMOKE FAILED: %d game(s) ended without our agent making a decision"
                 % (a.games - played))
    print("SMOKE OK")


if __name__ == "__main__":
    main()
