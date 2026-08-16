"""Is a search playout deterministic given the determinization?

If search_begin fixes the deck ORDER (not just the multiset), then repeating a playout from
one root is a no-op and n_playouts is wasted. The fix would be many determinizations
(shuffled pools), each branching all K candidates -- common random numbers WITHIN a scenario,
averaged ACROSS scenarios.
"""
import collections, os, random, sys
ROOT = "/root/ptcg/repo"
for p in (ROOT, os.path.join(ROOT, "cg-lib"), os.path.join(ROOT, "tools")):
    if p not in sys.path: sys.path.insert(0, p)
import library, rl_branch
import cg.api as api
from cg.game import battle_start, battle_select, battle_finish
from lm.agent import make_lm_agent

pilot, oppn = "alakazam", "dragapult"
d_me, d_op = library.read_deck(pilot), library.read_deck(oppn)
a_me, a_op = make_lm_agent(pilot, None, None), make_lm_agent(oppn, None, None)
obs, _ = battle_start(d_me, d_op)
tested = 0
res_same_root, res_reshuffled = [], []
try:
    for _ in range(4000):
        cur = obs.get("current")
        if cur is None or cur.get("result",-1)!=-1 or obs.get("select") is None: break
        sel = obs["select"]; opts = sel.get("option") or []
        if (cur["yourIndex"]==0 and len(opts)>=2 and sel.get("minCount",1)==1
                and sel.get("maxCount",1)==1 and tested < 6):
            try:
                mu, ou = rl_branch.unseen_multisets(obs, d_me, d_op)
            except rl_branch.DeterminizationError:
                obs = battle_select(a_me(obs)); continue
            tested += 1
            # A: same root, one candidate, 8 playouts
            o = api.to_observation_class(obs)
            root = api.search_begin(o, mu, mu, ou, ou, ou, [])
            outs = []
            for _i in range(8):
                st = rl_branch._raw_step(root.searchId, [0])
                if st.get("error",0)==0 and st.get("state"):
                    outs.append(rl_branch._playout(st["state"], 0, a_me, a_op))
            api.search_end()
            res_same_root.append(len(set(outs)))
            # B: fresh determinization (shuffled pool) each time, same candidate
            outs2 = []
            rng = random.Random(7)
            for _i in range(8):
                m2, o2 = list(mu), list(ou)
                rng.shuffle(m2); rng.shuffle(o2)
                r2 = api.search_begin(api.to_observation_class(obs), m2, m2, o2, o2, o2, [])
                st = rl_branch._raw_step(r2.searchId, [0])
                if st.get("error",0)==0 and st.get("state"):
                    outs2.append(rl_branch._playout(st["state"], 0, a_me, a_op))
                api.search_end()
            res_reshuffled.append(len(set(outs2)))
        obs = battle_select((a_me if cur["yourIndex"]==0 else a_op)(obs))
finally:
    battle_finish()
print("branch points tested:", tested)
print("A same root, 8 playouts   -> distinct outcomes per point:", res_same_root)
print("B reshuffled pool x8      -> distinct outcomes per point:", res_reshuffled)
print()
print("A all-identical at %d/%d points" % (sum(1 for x in res_same_root if x==1), len(res_same_root)))
print("B all-identical at %d/%d points" % (sum(1 for x in res_reshuffled if x==1), len(res_reshuffled)))
