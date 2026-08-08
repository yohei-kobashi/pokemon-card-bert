#!/usr/bin/env python3
"""Mirror self-play rollouts at temperature, for the shaped-return policy gradient.

NO engine_v2 IN THE TRAJECTORY. Both seats are the policy; the same decklist and the same
shuffle order sit on both sides, so two games from one seed differ only in what the policy
chose. `make_lm_agent` still constructs engine_v2 as its fallback -- that fires only when the
scorer refuses a menu, and every such decision is counted and marked `fallback` so the trainer
can drop it. If that count is not near zero, the rollout is not on-policy and must be fixed
before training, not trained around.

HOW SAMPLING IS DONE. The agent takes argmax over the scorer's output and then maps the index
back through its own dedup. Re-implementing that mapping here is how a silent off-by-one gets
in ([[rerank-data-label-bugs]] is the precedent), so instead the scorer is wrapped: it returns
a vector whose ARGMAX IS THE SAMPLED CANDIDATE. The agent's dedup, multi-pick unrolling and
index mapping all run untouched, and the true scores are recorded for the gradient.

    PYTHONPATH=cg-lib python3 tools/rl_rollout.py --model /root/out/d41_r8 \\
        --deck dragapult_dusknoir --seeds 8 --group 4 --temp 1.0 --out /root/rl/roll_1.jsonl.gz
"""

import argparse
import collections
import gzip
import json
import math
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "cg-lib"), os.path.join(ROOT, "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

ITCHY_POLLEN = 323          # Budew: the item lock, which the observation does not expose


def _softmax(xs, t):
    m = max(xs)
    e = [math.exp((x - m) / max(1e-6, t)) for x in xs]
    s = sum(e) or 1.0
    return [x / s for x in e]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="reranker checkpoint dir")
    ap.add_argument("--deck", default="dragapult_dusknoir")
    ap.add_argument("--seeds", type=int, default=8, help="distinct deals")
    ap.add_argument("--seed-base", type=int, default=700000)
    ap.add_argument("--group", type=int, default=4, help="trajectories per deal")
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=4000)
    ap.add_argument("--mirror-so", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import library
    from lm.agent import make_lm_agent
    from mirror_env import DEFAULT_SO, MirrorEngine, play
    from mirror_match import HFRerankScorer
    from dusk_potential import phi
    import rl_config

    fmt = dict(rl_config.PROMPT_FMT)
    ids = [int(x) for x in open(library.deck_path(a.deck)) if x.strip()]
    tuning = json.load(open(os.path.join(ROOT, "agents", "tuning.json")))
    prof = tuning.get(a.deck, {})
    sc = HFRerankScorer(a.model)
    eng = MirrorEngine(a.mirror_so or DEFAULT_SO)

    rng = random.Random(0)
    rec = []                     # decisions of the CURRENT trajectory
    state = {"lock": [False, False], "fallback": 0, "scored": 0, "trivial": 0}

    orig_score = sc.score

    def tapped(prompt, cands, obs=None):
        s = orig_score(prompt, cands, obs)
        if not s or len(s) != len(cands):
            return s
        p = _softmax(list(s), a.temp)
        j = rng.random()
        acc, pick = 0.0, len(p) - 1
        for i, q in enumerate(p):
            acc += q
            if j <= acc:
                pick = i
                break
        cur = (obs or {}).get("current") or {}
        yi = cur.get("yourIndex", 0)
        rec.append({"seat": yi, "turn": cur.get("turn"),
                    "prompt": prompt, "cands": list(cands),
                    "chosen": pick, "scores": [round(float(x), 5) for x in s],
                    "phi": round(phi(obs, yi, lock=state["lock"][yi]), 4)})
        state["scored"] += 1
        # Rewrite so the agent's own argmax lands on the sample. Its dedup and index mapping
        # then run exactly as in evaluation.
        out = [0.0] * len(s)
        out[pick] = 1.0
        return out
    sc.score = tapped

    agent = make_lm_agent(ids, prof, model=sc, deck_name=a.deck, **fmt)

    def watching_agent(obs):
        before = state["scored"]
        n_opt = len((obs.get("select") or {}).get("option") or [])
        pick = agent(obs)
        if state["scored"] == before and n_opt >= 2:
            # Only a REAL choice counts as contamination. A one-option menu never reaches the
            # scorer and has nothing to learn from either; counting those put the fallback rate
            # at 12.4% when the policy had in fact answered every decision that had a choice.
            state["fallback"] += 1
        state["trivial"] += 1 if n_opt < 2 else 0
        # Budew's Itchy Pollen locks the OPPONENT's items next turn, and no observation field
        # reports it -- verified against a real board. We know because we chose it.
        try:
            sel = obs.get("select") or {}
            opts = sel.get("option") or []
            cur = obs.get("current") or {}
            yi = cur.get("yourIndex", 0)
            for i in (pick if isinstance(pick, (list, tuple)) else [pick]):
                o = opts[i] if isinstance(i, int) and 0 <= i < len(opts) else None
                if isinstance(o, dict) and o.get("attackId") == ITCHY_POLLEN:
                    state["lock"][1 - yi] = True
        except Exception:                                   # noqa: BLE001
            pass
        return pick

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    t0 = time.time()
    n_dec = n_games = 0
    agree = collections.Counter()
    with gzip.open(a.out, "wt") as f:
        f.write(json.dumps({"header": 1, "model": a.model, "deck": a.deck,
                            "temp": a.temp, "group": a.group}) + "\n")
        for si in range(a.seeds):
            seed = a.seed_base + si
            group_actions = []
            for k in range(a.group):
                del rec[:]
                state["lock"] = [False, False]
                state["fallback"] = 0
                state["trivial"] = 0
                rng.seed(seed * 1000 + k)          # the deal is fixed; only sampling varies
                r = play(eng, watching_agent, watching_agent, ids, ids, seed,
                         mirror=1, max_steps=a.max_steps)
                eng.finish()
                n_games += 1
                n_dec += len(rec)
                group_actions.append([d["chosen"] for d in rec])
                f.write(json.dumps({"seed": seed, "k": k, "result": r,
                                    "fallback": state["fallback"],
                                    "trivial": state["trivial"],
                                    "decisions": rec}) + "\n")
            # STEP 3 OF THE LADDER: if the group does not diverge there is no gradient, and
            # every advantage in it is exactly 0.
            m = min(len(x) for x in group_actions)
            for t in range(m):
                vals = {x[t] for x in group_actions}
                agree["same" if len(vals) == 1 else "diff"] += 1
            print("  seed %d | %d games | %d decisions | %.0fs"
                  % (seed, a.group, n_dec, time.time() - t0), flush=True)

    tot = agree["same"] + agree["diff"]
    print("\n%d games, %d decisions -> %s" % (n_games, n_dec, a.out))
    print("group divergence: %d/%d decisions differ within a group (%.1f%%)"
          % (agree["diff"], tot, 100.0 * agree["diff"] / max(1, tot)))
    if tot and agree["diff"] / tot < 0.30:
        print("WARNING: below the 30%% the design calls for -- raise --temp or the gradient "
              "is mostly zeros")


if __name__ == "__main__":
    main()
