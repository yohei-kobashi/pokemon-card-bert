"""Measure a submission bundle against the LIVE time budget, per game.

The question this answers: does a game finish inside Kaggle's ~600 s, and where does the time go?
The previous LM submission (rr_v37_dragapult, 2026-07-28) came back SubmissionStatus.ERROR while
its own description claimed "real-game bank max 154s/600s over 6 games" -- so either that
measurement or the deployment is wrong, and a bundle cannot be resubmitted until the numbers
agree.

Reports per game: wall time, time consumed inside the scorer (its own bank), how many decisions
were scored by the LM vs handed to the engine_v2 fallback after the bank ran out, and the
per-decision latency distribution. Threads are pinned to 4, the competition's vCPU count -- a
64-core dev box otherwise flatters every number.

Run:  python time_bundle.py <bundle.tar.gz> <deck> [games]
"""
import os
import subprocess
import sys
import tempfile
import time

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "4"

TAR = sys.argv[1]
DECK = sys.argv[2]
GAMES = int(sys.argv[3]) if len(sys.argv) > 3 else 1
REPO = "/root/ptcg/repo"


def main():
    work = tempfile.mkdtemp(prefix="timing_")
    subprocess.run(["tar", "xzf", TAR, "-C", work], check=True)
    e = os.listdir(work)
    inner = os.path.join(work, e[0]) if len(e) == 1 and os.path.isdir(os.path.join(work, e[0])) \
        else work
    sys.path.insert(0, inner)
    os.chdir(inner)

    t_import = time.time()
    import importlib.util
    spec = importlib.util.spec_from_file_location("subm_main", os.path.join(inner, "main.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    t_import = time.time() - t_import
    sc = getattr(m, "_scorer", None)
    print("import (model load) %.1fs | scorer=%s" % (t_import, type(sc).__name__ if sc else None))
    if sc is None:
        raise SystemExit("fell back to engine_v2 at import")

    lat = []
    orig = sc.score

    def wrapped(prompt, cands, obs=None):
        t = time.time()
        try:
            return orig(prompt, cands, obs)
        finally:
            lat.append(time.time() - t)
    sc.score = wrapped

    sys.path.insert(0, REPO)
    sys.path.insert(0, os.path.join(REPO, "cg-lib"))
    import json
    import library
    from cg.game import battle_start, battle_select, battle_finish
    from lm.agent import make_lm_agent
    tun = json.load(open(os.path.join(REPO, "agents", "tuning.json")))
    opp_name = "alakazam" if DECK != "alakazam" else "crustle"
    d_me, d_op = library.read_deck(DECK), library.read_deck(opp_name)
    opp = make_lm_agent(opp_name, tun.get(opp_name), None)

    for g in range(GAMES):
        lat.clear()
        spent0 = getattr(sc, "spent", 0.0)
        t0 = time.time()
        obs, _ = battle_start(d_me, d_op)
        n_dec = 0
        try:
            for _ in range(4000):
                cur = obs.get("current")
                if cur is None or cur.get("result", -1) != -1:
                    break
                if obs.get("select") is None:
                    break
                if cur["yourIndex"] == 0:
                    n_dec += 1
                    pick = m.agent(obs)
                else:
                    pick = opp(obs)
                obs = battle_select(pick)
        finally:
            try:
                battle_finish()
            except Exception:
                pass
        wall = time.time() - t0
        spent = getattr(sc, "spent", 0.0) - spent0
        L = sorted(lat)
        p = lambda f: L[int(f * (len(L) - 1))] if L else 0.0
        print("game %d: WALL %6.1fs | my decisions %3d | LM-scored %3d (%.0f%%) | "
              "scorer spent %6.1fs | latency mean %.2f p50 %.2f p90 %.2f max %.2f"
              % (g, wall, n_dec, len(L), 100.0 * len(L) / max(1, n_dec), spent,
                 sum(L) / max(1, len(L)), p(.5), p(.9), p(1.0)))
        print("        VERDICT vs a 600s budget: %s" % ("OVER" if wall > 600 else "ok"))


if __name__ == "__main__":
    main()
