"""Probe 2: can engine_v2 drive a search-tree playout?

api.py converts the native JSON into dataclasses, but engine_v2 (and every agent in the
repo) consumes the raw dict observation. If the native SearchStep JSON already carries a
dict-shaped observation we can bypass the dataclass layer entirely and reuse engine_v2 as
the playout policy with no adapter.

Checks:
  A. raw SearchBegin/SearchStep JSON shape -- is `state.observation` the same dict an
     agent normally sees (keys: select / logs / current)?
  B. does engine_v2 accept it and return a legal selection?
  C. full playout to terminal DRIVEN BY engine_v2 on both sides, timed.
"""
import ctypes
import json
import os
import sys
import time

ROOT = "/root/ptcg/repo"
for p in (ROOT, os.path.join(ROOT, "cg-lib"), os.path.join(ROOT, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

import library                                                    # noqa: E402
import cg.api as api                                              # noqa: E402
import cg.sim as sim                                              # noqa: E402
from cg.game import battle_start, battle_select, battle_finish     # noqa: E402
from lm.agent import make_lm_agent                                 # noqa: E402

DECK_A, DECK_B = "alakazam", "dragapult"


def sel_first(sel):
    if not sel:
        return []
    n, lo, hi = len(sel["option"]), sel.get("minCount", 1), sel.get("maxCount", 1)
    k = min(max(lo, min(hi, 1)), n)
    return list(range(k)) if k > 0 else []


def raw_step(search_id, select):
    """lib.SearchStep without api.py's dataclass conversion -> plain dict."""
    arr = (ctypes.c_int * len(select))(*select)
    bs = sim.lib.SearchStep(api.agent_ptr, search_id, arr, len(select))
    return json.loads(bs.decode())


def main():
    dl = library.read_deck(DECK_A)
    ol = library.read_deck(DECK_B)
    obs, _ = battle_start(dl, ol)
    try:
        for step in range(400):
            cur = obs["current"]
            if cur.get("result", -1) != -1:
                print("ended during setup; rerun")
                return
            if step >= 8 and len(obs.get("select", {}).get("option") or []) >= 3:
                break
            obs = battle_select(sel_first(obs.get("select")))

        o = api.to_observation_class(obs)
        root = api.search_begin(o, dl, dl, ol, ol, ol, [])
        print("root id", root.searchId)

        # --- A: raw JSON shape -------------------------------------------------
        raw = raw_step(root.searchId, sel_first(obs["select"]))
        print("\n--- A: raw SearchStep JSON ---")
        print("  top-level keys:", list(raw.keys()))
        st = raw.get("state") or {}
        print("  state keys:", list(st.keys()))
        ro = st.get("observation")
        print("  observation keys:", list(ro.keys()) if isinstance(ro, dict) else type(ro))
        if isinstance(ro, dict) and "current" in ro:
            print("  current keys:", list(ro["current"].keys()))
            print("  SHAPE MATCHES A NORMAL OBS:",
                  set(["select", "current"]).issubset(set(ro.keys())))

        # --- B: engine_v2 on a search observation ------------------------------
        print("\n--- B: engine_v2 on the search observation ---")
        eng_pilot = make_lm_agent(DECK_A, None, None)   # pure engine_v2
        eng_opp = make_lm_agent(DECK_B, None, None)
        try:
            pick = eng_pilot(ro)
            print("  engine_v2 returned:", pick, " (legal length:",
                  len(ro.get("select", {}).get("option") or []), "options)")
        except Exception as e:
            print("  engine_v2 FAILED:", type(e).__name__, e)
            return

        # --- C: full engine_v2-driven playout to terminal -----------------------
        print("\n--- C: engine_v2-driven playout to terminal ---")
        cur_state = st
        steps = 0
        t0 = time.time()
        result = None
        pilot_i = obs["current"]["yourIndex"]
        for _ in range(4000):
            ob = cur_state["observation"]
            c = ob.get("current")
            if c is None:
                print("  current None at", steps)
                break
            if c.get("result", -1) != -1:
                result = c["result"]
                break
            if not ob.get("select"):
                print("  select None at", steps)
                break
            yi = c["yourIndex"]
            agent = eng_pilot if yi == pilot_i else eng_opp
            try:
                choice = agent(ob)
            except Exception as e:
                print("  agent raised at step %d: %s" % (steps, e))
                break
            nxt = raw_step(cur_state["searchId"], choice)
            if nxt.get("error", 0) != 0:
                print("  SearchStep error %s at step %d" % (nxt.get("error"), steps))
                break
            cur_state = nxt["state"]
            steps += 1
        dt = time.time() - t0
        print("  steps %d, terminal=%s (result=%s), %.3f s" % (steps, result is not None,
                                                              result, dt))
        if steps:
            print("  %.4f s/step -> a playout of this length = %.3f s (%.1f playouts/s/core)"
                  % (dt / steps, dt, 1.0 / max(dt, 1e-9)))
        api.search_end()
    finally:
        battle_finish()


if __name__ == "__main__":
    main()
