"""Would the allocator have chosen better than the policy did? Replay the recorded spread
decisions and compare, using the plan's own rule as the yardstick."""
import gzip, json, sys, collections
sys.path.insert(0, "/root/ptcg/repo"); sys.path.insert(0, "/root/ptcg/repo/cg-lib")
sys.path.insert(0, "/root/ptcg/repo/tools")
import library
from mirror_env import MirrorEngine
from dusk_plan import opportunities
from dusk_spread import choose, is_spread_select

SO = "/root/ptcg/repo/data/kaggle_engine_ext/libcg_mirror.so"
eng = MirrorEngine(SO)
ids = {}
def d_ids(n):
    if n not in ids:
        ids[n] = [int(x) for x in open(library.deck_path(n)) if x.strip()]
    return ids[n]

DECK = "dragapult_dusknoir"
c = collections.Counter()
for path in sys.argv[1:]:
    for line in gzip.open(path, "rt"):
        d = json.loads(line)
        if d.get("header"):
            continue
        d0 = d.get("deck0") or d.get("deck"); d1 = d.get("deck1") or d.get("deck")
        if DECK not in (d0, d1):
            continue
        seat = 0 if d0 == DECK else 1
        obs = eng.start(d_ids(d0), d_ids(d1), d["seed"], mirror=1)
        try:
            for pick in d["picks"]:
                if obs is None:
                    break
                cur = obs.get("current") or {}
                if cur.get("result", -1) != -1 or obs.get("select") is None:
                    break
                if cur.get("yourIndex") == seat and is_spread_select(obs):
                    live = opportunities(obs, seat).get("spread_aim")
                    mine = choose(obs, seat)
                    took = set(pick if isinstance(pick, (list, tuple)) else [pick])
                    c["decisions"] += 1
                    if live:
                        c["rule_live"] += 1
                        c["policy_ok"] += 1 if (live & took) else 0
                        if mine is None:
                            c["allocator_declined"] += 1
                        else:
                            c["allocator_ok"] += 1 if mine in live else 0
                    else:
                        c["no_conformant_target"] += 1
                if pick is None:
                    break
                obs = eng.select(pick)
        finally:
            eng.finish()
print("spread decisions %d | rule live %d | nothing conformant %d"
      % (c["decisions"], c["rule_live"], c["no_conformant_target"]))
if c["rule_live"]:
    print("  policy chose a conformant target : %5.1f%%" % (100 * c["policy_ok"] / c["rule_live"]))
    print("  allocator chose one              : %5.1f%%  (declined %d)"
          % (100 * c["allocator_ok"] / c["rule_live"], c["allocator_declined"]))
