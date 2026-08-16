"""Same yardstick as the spread check: on the recorded attach decisions, how often does the
policy pick a plan-conformant target, and how often does the deterministic rule?"""
import gzip, json, sys, collections
for p in ("/root/ptcg/repo", "/root/ptcg/repo/cg-lib", "/root/ptcg/repo/tools"):
    sys.path.insert(0, p)
import library
from mirror_env import MirrorEngine
from dusk_plan import opportunities
from dusk_energy import choose, is_attach_select

eng = MirrorEngine("/root/ptcg/repo/data/kaggle_engine_ext/libcg_mirror.so")
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
                if cur.get("yourIndex") == seat and is_attach_select(obs):
                    live = opportunities(obs, seat)
                    took = set(pick if isinstance(pick, (list, tuple)) else [pick])
                    mine = choose(obs, seat)
                    for rule in ("energy_line", "energy_focus"):
                        s = live.get(rule)
                        if not s:
                            continue
                        c[rule + "_n"] += 1
                        c[rule + "_policy"] += 1 if (s & took) else 0
                        if mine is None:
                            c[rule + "_declined"] += 1
                        else:
                            c[rule + "_rule"] += 1 if mine in s else 0
                if pick is None:
                    break
                obs = eng.select(pick)
        finally:
            eng.finish()
for rule in ("energy_line", "energy_focus"):
    n = c[rule + "_n"]
    if not n:
        print("%s: no opportunities" % rule); continue
    print("%s  (%d chances)" % (rule, n))
    print("   policy conformant : %5.1f%%" % (100 * c[rule + "_policy"] / n))
    print("   rule   conformant : %5.1f%%   (declined %d)"
          % (100 * c[rule + "_rule"] / n, c[rule + "_declined"]))
