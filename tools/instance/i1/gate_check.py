"""When the rules have nothing left to say, is there genuinely nothing left to DO?

The proposal "if no rule applies, attack" is only safe if the gate opens on menus that offer
nothing but the attack. The rows in plan_r4 cannot answer this -- their scope was trimmed to
attacks+end+retreat, so the rest of the menu is not in the file. Replay and look at the whole
menu at the moment the gate opens."""
import gzip, json, sys, collections
for p in ("/root/ptcg/repo", "/root/ptcg/repo/cg-lib", "/root/ptcg/repo/tools"):
    sys.path.insert(0, p)
import library
from mirror_env import MirrorEngine
from lm.actions import encode_option
from dusk_plan import opportunities

eng = MirrorEngine("/root/ptcg/repo/data/kaggle_engine_ext/libcg_mirror.so")
ids = {}
def d(n):
    if n not in ids:
        ids[n] = [int(v) for v in open(library.deck_path(n)) if v.strip()]
    return ids[n]

DECK = "dragapult_dusknoir"
c = collections.Counter()
rest = collections.Counter()
for path in sys.argv[1:]:
    for line in gzip.open(path, "rt"):
        g = json.loads(line)
        if g.get("header"):
            continue
        d0 = g.get("deck0") or g.get("deck"); d1 = g.get("deck1") or g.get("deck")
        if DECK not in (d0, d1):
            continue
        seat = 0 if d0 == DECK else 1
        obs = eng.start(d(d0), d(d1), g["seed"], mirror=1)
        try:
            for pick in g["picks"]:
                if obs is None:
                    break
                cur = obs.get("current") or {}
                if cur.get("result", -1) != -1 or obs.get("select") is None:
                    break
                if cur.get("yourIndex") == seat:
                    live = opportunities(obs, seat)
                    if live and set(live) <= {"phantom_dive", "lock_early"}:
                        # the gate is OPEN: rules have nothing left to say except "attack"
                        c["gate_open"] += 1
                        opts = (obs.get("select") or {}).get("option") or []
                        ks = set()
                        for o in opts:
                            try:
                                ks.add(encode_option(o, obs).split(":")[0])
                            except Exception:      # noqa: BLE001
                                pass
                        other = ks - {"attack", "end", "retreat"}
                        if other:
                            c["still_has_plays"] += 1
                            for k in other:
                                rest[k] += 1
                        else:
                            c["truly_nothing_left"] += 1
                if pick is None:
                    break
                obs = eng.select(pick)
        finally:
            eng.finish()
n = c["gate_open"] or 1
print("gate opened (only attack rules live): %d" % c["gate_open"])
print("  truly nothing else on the menu : %5d  (%.1f%%)" % (c["truly_nothing_left"], 100*c["truly_nothing_left"]/n))
print("  still has non-attack options   : %5d  (%.1f%%)" % (c["still_has_plays"], 100*c["still_has_plays"]/n))
print("  what those options are:", dict(rest.most_common()))
