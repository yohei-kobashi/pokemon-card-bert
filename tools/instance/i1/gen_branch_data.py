"""Generate branch-point training data: prompt + candidates + 8-playout Q. CPU only.

Purpose. `tools/rank_probe.py` decides whether the RL plateau is an OPTIMISATION limit or a
REPRESENTATION limit by fitting the cross-encoder SUPERVISED to rank candidates by their playout
value. But it reads the rollout's recorded `qvals`, which are 2-playout Q in {-1,0,+1} -- ties
everywhere. A supervised fit can fail on labels that noisy for reasons that have nothing to do
with representation, which would produce a FALSE "representation limit" verdict. So: regenerate
the labels at 8 playouts. The same data is the training set for the search-distillation method,
so this is not throwaway work.

Output records are the SAME SHAPE rank_probe already reads -- `prompt`, `cands`, `qvals` (None
for un-branched candidates) -- plus `qsel`/`qval` (the 4/4 playout split, for an unbiased
selection-vs-scoring metric) and `nplay` per candidate.

The prompt is built with `rl_rollout.make_serializer`, NOT a local call: passing deck_ids and
deck_name is what renders `DECK[...]` and `ID ME d_x a_y`, and a bare serialize_stateless() call
silently drops both (that is [[bundle-drops-id-segment]], and I re-made the same mistake while
writing this file).

STATES ARE ENGINE_V2-PILOTED, not policy-piloted -- the GPU is busy and the policy cannot run.
That matches how `attach-decisions-at-chance` was measured, so it is a fair test of "can the
model represent this ranking", but it is NOT the policy's own state distribution. For the
distillation method proper, regenerate on-policy once the GPU is free.

Run:  CUDA_VISIBLE_DEVICES="" python gen_branch_data.py OUT.jsonl.gz [games_per_pair] [workers]
"""
import gzip
import json
import os
import random
import sys

ROOT = "/root/ptcg/repo"
for p in (ROOT, os.path.join(ROOT, "cg-lib"), os.path.join(ROOT, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

OUT = sys.argv[1] if len(sys.argv) > 1 else "/root/out/branch8.jsonl.gz"
GAMES_PER_PAIR = int(sys.argv[2]) if len(sys.argv) > 2 else 240
WORKERS = int(sys.argv[3]) if len(sys.argv) > 3 else 120

K = 6                 # candidates branched per point (a random subset -- no policy involved)
NPLAY = 8             # playouts per candidate, split 4 rank / 4 score
PER_GAME = 20         # cap per game
ACCEPT = 0.30         # thin sampling so points spread over the whole game

PILOTS = ["alakazam", "crustle", "dragapult", "dragapult_dusknoir",
          "marnie_grimmsnarl", "rockets_honchkrow", "rockets_mewtwo"]
OPPS = ["alakazam", "crustle", "dragapult"]
PAIRS = [(p, o) for p in PILOTS for o in OPPS]


def one_game(task):
    pilot, opp, seed = task
    import library
    import rl_branch
    import rl_rollout
    import cg.api as api
    from cg.game import battle_start, battle_select, battle_finish
    from lm.agent import make_lm_agent
    from lm.actions import encode_option

    rng = random.Random(seed)
    try:
        d_me, d_op = library.read_deck(pilot), library.read_deck(opp)
    except Exception:
        return []
    a_me = make_lm_agent(pilot, None, None)     # model=None -> engine_v2
    a_op = make_lm_agent(opp, None, None)
    ser = rl_rollout.make_serializer(d_me, pilot)
    pilot_i = seed % 2
    d0, d1 = (d_me, d_op) if pilot_i == 0 else (d_op, d_me)
    obs, _ = battle_start(d0, d1)
    if obs is None:
        return []
    recs, taken = [], 0
    try:
        for _ in range(4000):
            cur = obs.get("current")
            if cur is None or cur.get("result", -1) != -1:
                break
            sel = obs.get("select")
            if sel is None:
                break
            yi = cur["yourIndex"]
            opts = sel.get("option") or []
            eligible = (yi == pilot_i and len(opts) >= 2
                        and sel.get("minCount", 1) == 1 and sel.get("maxCount", 1) == 1)
            if eligible and taken < PER_GAME and rng.random() < ACCEPT:
                r = _point(obs, cur, opts, pilot, opp, pilot_i, d_me, d_op, a_me, a_op,
                           rng, ser, rl_branch, api, encode_option)
                if r is not None:
                    recs.append(r)
                    taken += 1
            obs = battle_select((a_me if yi == pilot_i else a_op)(obs))
    except Exception:
        pass
    finally:
        try:
            battle_finish()
        except Exception:
            pass
    return recs


def _point(obs, cur, opts, pilot, opp, pilot_i, d_me, d_op, a_me, a_op,
           rng, ser, rl_branch, api, encode_option):
    try:
        mu, ou = rl_branch.unseen_multisets(obs, d_me, d_op)
    except Exception:
        return None                       # determinization did not reconcile (~3.6%)
    n = len(opts)
    idx = sorted(rng.sample(range(n), min(K, n)))
    outcomes = [[] for _ in idx]
    for _rep in range(NPLAY):
        m2, o2 = list(mu), list(ou)
        rng.shuffle(m2)
        rng.shuffle(o2)
        try:
            root = api.search_begin(api.to_observation_class(obs), m2, m2, o2, o2, o2, [])
        except Exception:
            continue
        try:
            for j, k in enumerate(idx):
                st = rl_branch._raw_step(root.searchId, [k])
                if st.get("error", 0) != 0 or not st.get("state"):
                    continue
                v = rl_branch._playout(st["state"], pilot_i, a_me, a_op)
                if v is not None:
                    outcomes[j].append(v)
        finally:
            api.search_end()
    half = NPLAY // 2
    if not all(len(v) >= half + 1 for v in outcomes):
        return None                       # need both halves populated
    qv = [None] * n
    qs = [None] * n
    qa = [None] * n
    npl = [None] * n
    for j, k in enumerate(idx):
        v = outcomes[j]
        qv[k] = sum(v) / len(v)
        qs[k] = sum(v[:half]) / len(v[:half])
        qa[k] = sum(v[half:]) / len(v[half:])
        npl[k] = len(v)
    return dict(matchup="%s__vs__%s" % (pilot, opp),
                prompt=rl_rollout_ACT() + ser(obs),
                cands=[encode_option(o, obs) for o in opts],
                qvals=qv, qsel=qs, qval=qa, nplay=npl,
                turn=cur.get("turn", -1),
                my_prizes=len(cur["players"][pilot_i].get("prize") or []),
                op_prizes=len(cur["players"][1 - pilot_i].get("prize") or []))


def rl_rollout_ACT():
    import rl_rollout
    return rl_rollout._ACT


def main():
    tasks = [(p, o, s) for (p, o) in PAIRS for s in range(GAMES_PER_PAIR)]
    random.Random(0).shuffle(tasks)
    print("gen_branch_data: %d games over %d pairs, K=%d, %d playouts, %d workers -> %s"
          % (len(tasks), len(PAIRS), K, NPLAY, WORKERS, OUT), flush=True)
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    import multiprocessing as mp
    n = 0
    with gzip.open(OUT, "wt") as f, mp.Pool(WORKERS) as pool:
        for i, rr in enumerate(pool.imap_unordered(one_game, tasks, chunksize=1)):
            for r in rr:
                f.write(json.dumps(r) + "\n")
                n += 1
            if (i + 1) % 200 == 0:
                print("  %d/%d games, %d branch points" % (i + 1, len(tasks), n), flush=True)
    print("\nwrote %s: %d branch points" % (OUT, n))
    print("size: %.1f MB" % (os.path.getsize(OUT) / 1e6))


if __name__ == "__main__":
    main()
