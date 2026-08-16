"""Probe the native search tree for decision-level RL grouping.

Three unknowns left after confirming K-way branching works:
  1. can a branch be stepped all the way to a TERMINAL result? (a decision-level
     advantage needs a win/loss at the end of each branch)
  2. how fast is search_step, relative to engine_v2's 22 games/s/core?
  3. does search_begin's root reproduce the REAL visible state? (search_begin only
     validates card COUNTS, so "it returned a node" is not evidence of correctness)

Run on the vast box:  CUDA_VISIBLE_DEVICES="" python /root/probe_search.py
"""
import os
import sys
import time

ROOT = "/root/ptcg/repo"
for p in (ROOT, os.path.join(ROOT, "cg-lib"), os.path.join(ROOT, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

import library                                    # noqa: E402
import cg.api as api                              # noqa: E402
from cg.game import battle_start, battle_select, battle_finish   # noqa: E402

DECK_A, DECK_B = "alakazam", "dragapult"


def sel_first(sel):
    """First legal selection, for a dict-shaped or dataclass-shaped select."""
    if sel is None:
        return []
    if isinstance(sel, dict):
        n, lo, hi = len(sel["option"]), sel["minCount"], sel["maxCount"]
    else:
        n, lo, hi = len(sel.option), sel.minCount, sel.maxCount
    k = min(max(lo, min(hi, 1)), n)
    return list(range(k)) if k > 0 else []


def n_options(sel):
    if sel is None:
        return 0
    return len(sel["option"] if isinstance(sel, dict) else sel.option)


def advance_to_midgame(obs, min_opts=3, min_step=8, max_step=400):
    """Play the REAL game forward until we are past setup and have a real choice."""
    for step in range(max_step):
        cur = obs["current"]
        if cur.get("result", -1) != -1:
            return obs, step, True
        if step >= min_step and n_options(obs.get("select")) >= min_opts:
            return obs, step, False
        obs = battle_select(sel_first(obs.get("select")))
    return obs, max_step, False


def dump_player_struct(obs):
    print("--- player struct (fields available for building a determinization) ---")
    cur = obs["current"]
    yi = cur["yourIndex"]
    for who, idx in (("me ", yi), ("opp", 1 - yi)):
        p = cur["players"][idx]
        bits = []
        for k, v in p.items():
            if isinstance(v, list):
                bits.append("%s[%d]" % (k, len(v)))
            else:
                bits.append("%s=%s" % (k, v))
        print("  %s  %s" % (who, "  ".join(bits)))
    return yi


def visible_fingerprint(state, yi):
    """Visible-only summary, comparable between the real obs and a search root."""
    def g(o, k, d=None):
        if isinstance(o, dict):
            return o.get(k, d)
        return getattr(o, k, d)

    players = g(state, "players")
    out = {"turn": g(state, "turn"), "yourIndex": g(state, "yourIndex")}
    for tag, idx in (("me", yi), ("opp", 1 - yi)):
        p = players[idx]
        out[tag] = {
            "deckCount": g(p, "deckCount"),
            "handCount": g(p, "handCount"),
            "nPrize": len(g(p, "prize") or []),
            "nBench": len(g(p, "bench") or []),
            "nDiscard": len(g(p, "discard") or []),
            "nActive": len(g(p, "active") or []),
        }
    return out


def main():
    dl = library.read_deck(DECK_A)
    ol = library.read_deck(DECK_B)
    obs, _sd = battle_start(dl, ol)
    try:
        obs, step, ended = advance_to_midgame(obs)
        if ended:
            print("game ended during setup -- rerun")
            return
        cur = obs["current"]
        print("real game: step %d, turn %d, options %d"
              % (step, cur["turn"], n_options(obs.get("select"))))
        yi = dump_player_struct(obs)

        o = api.to_observation_class(obs)
        t0 = time.time()
        root = api.search_begin(o, dl, dl, ol, ol, ol, [])
        t_begin = time.time() - t0
        print("\nsearch_begin: %.4f s, root id %d" % (t_begin, root.searchId))

        # --- TEST 3: does the root reproduce the real visible state? -------------
        print("\n--- TEST 3: root vs real visible state ---")
        real_fp = visible_fingerprint(cur, yi)
        root_fp = visible_fingerprint(root.observation.current, yi)
        same = real_fp == root_fp
        print("  real: %s" % real_fp)
        print("  root: %s" % root_fp)
        print("  MATCH: %s" % same)
        if not same:
            for k in real_fp:
                if real_fp[k] != root_fp.get(k):
                    print("    differs at %-10s real=%s root=%s"
                          % (k, real_fp[k], root_fp.get(k)))

        # --- TEST 1: can a branch reach a terminal result? -----------------------
        print("\n--- TEST 1: play one branch to terminal ---")
        node = api.search_step(root.searchId, sel_first(root.observation.select))
        steps = 0
        t0 = time.time()
        result = None
        for _ in range(4000):
            st = node.observation.current
            if st is None:
                print("  observation.current is None at step %d" % steps)
                break
            r = st.result if not isinstance(st, dict) else st.get("result", -1)
            if r != -1:
                result = r
                break
            if node.observation.select is None:
                print("  select is None at step %d (cannot continue)" % steps)
                break
            node = api.search_step(node.searchId, sel_first(node.observation.select))
            steps += 1
        dt = time.time() - t0
        print("  steps taken: %d" % steps)
        print("  TERMINAL REACHED: %s (result=%s)" % (result is not None, result))

        # --- TEST 2: speed --------------------------------------------------------
        print("\n--- TEST 2: speed ---")
        if steps:
            print("  %.5f s / search_step  (%.0f steps/s)" % (dt / steps, steps / dt))
            print("  a ~70-step playout would cost %.3f s -> %.1f playouts/s/core"
                  % (dt / steps * 70, 1.0 / (dt / steps * 70)))
        print("  (engine_v2 reference: 0.045 s/game = 22 games/s/core)")

        api.search_end()
    finally:
        battle_finish()


if __name__ == "__main__":
    main()
