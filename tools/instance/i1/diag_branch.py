import gzip, json, os, random, sys, traceback
sys.path.insert(0, "/root/ptcg/repo"); sys.path.insert(0, "/root/ptcg/repo/cg-lib")
sys.path.insert(0, "/root/ptcg/repo/tools")
import library
from mirror_env import MirrorEngine, DEFAULT_SO
from lm.agent import make_lm_agent
import rl_branch, cg.api as api

g = None
for line in gzip.open("/root/smoke_trace.jsonl.gz", "rt"):
    d = json.loads(line)
    if d.get("header"):
        continue
    if any(m[1] is not None for m in d["meta"]):
        g = d
        break
deck, seed = g["deck"], g["seed"]
tgt = next(t for t, m in enumerate(g["meta"]) if m[1] is not None and m[2] is not None)
print("deck %s seed %d target decision %d margin %s alt %s" % (deck, seed, tgt, g["meta"][tgt][1], g["meta"][tgt][2]))

ids = [int(x) for x in open(library.deck_path(deck)) if x.strip()]
prof = json.load(open("/root/ptcg/repo/agents/tuning.json")).get(deck, {})
me = make_lm_agent(ids, prof, model=None)
opp = make_lm_agent(ids, prof, model=None)

eng = MirrorEngine(DEFAULT_SO)
obs = eng.start(ids, ids, seed, mirror=1)
for t in range(tgt):
    obs = eng.select(g["picks"][t])
print("reached target. blob present:", bool(obs.get("search_begin_input")),
      "menu", len((obs.get("select") or {}).get("option") or []), "recorded nc", g["meta"][tgt][3])

pick, alt = g["picks"][tgt], g["meta"][tgt][2]
sels = [list(pick), [alt]]
rng = random.Random(1)

# manual scenario, with real tracebacks
mu, ou = rl_branch.unseen_multisets(obs, ids, ids)
mu, ou = list(mu), list(ou)
rng.shuffle(mu); rng.shuffle(ou)
cur = obs["current"]; yi = cur["yourIndex"]
oa = cur["players"][1 - yi].get("active") or []
ag = [ou[0]] if (len(oa) > 0 and oa[0] is None and ou) else []
try:
    root = api.search_begin(api.to_observation_class(obs), mu, mu, ou, ou, ou, ag)
    print("search_begin OK, searchId:", root.searchId)
    step = rl_branch._raw_step(root.searchId, sels[0])
    print("step error:", step.get("error", 0), "state:", bool(step.get("state")))
    v = rl_branch._playout(step["state"], 0, me, opp)
    print("playout value:", v)
    api.search_end()
except Exception:
    traceback.print_exc()

q = rl_branch.branch_values(obs, ids, ids, 0, sels, me, opp, n_playouts=4, rng=rng)
print("branch_values(4):", q)
eng.finish()
