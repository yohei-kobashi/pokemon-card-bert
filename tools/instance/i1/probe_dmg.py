import collections, json, os, sys
sys.path[:0] = [".", "cg-lib", "tools"]
import mirror_match as mm
import dusk_plan
from tools.mirror_env import DEFAULT_SO, MirrorEngine, play

eng = MirrorEngine(DEFAULT_SO)
tuning = json.load(open("agents/tuning.json"))
my_ids, opp_ids = mm.load_deck("dragapult_dusknoir"), mm.load_deck("ogerpon_mono")
spec = "planfilter:lethal_now,clops_hold,judge_timing,spare_ex_bench,retreat_energy,stadium_replace,search_bottom,setup_search,front_dive,promote_dive,promote_line,spread_kill,spread_reach,munki_close,denial_fuel:engine"
agent, _ = mm.make_agent(spec, "dragapult_dusknoir", my_ids, tuning.get("dragapult_dusknoir", {}))
opp, _ = mm.make_agent("engine", "ogerpon_mono", opp_ids, tuning.get("ogerpon_mono", {}))
seen = [0]
C = collections.Counter()
def watched(obs):
    sel = obs.get("select") or {}
    ctx = sel.get("context")
    C["menus"] += 1
    C["ctx_%s" % ctx] += 1
    if ctx in (13, 14):
        C["dmg_menus"] += 1
        if seen[0] < 8:
            seen[0] += 1
            cur = obs.get("current") or {}
            yi = cur.get("yourIndex", 0)
            opts = sel.get("option") or []
            print("DMG menu ctx=%s remain=%r nopts=%d" % (ctx, sel.get("remainDamageCounter"), len(opts)))
            for i, o in enumerate(opts[:8]):
                if isinstance(o, dict):
                    print("   ", i, {k: o.get(k) for k in ("playerIndex","inPlayArea","area","inPlayIndex","index","cardId","attackId")})
            live = dusk_plan.opportunities(obs)
            print("    fired:", {k: sorted(v[0]) for k, v in live.items() if k in ("spread_aim","spread_kill","spread_reach")})
    return agent(obs)
ES = {"current": None, "logs": [], "remainingOverageTime": 600.0, "search_begin_input": None, "select": None, "step": 1}
for i in range(40):
    watched(dict(ES)); opp(dict(ES))
    seed, mine = 500 + i // 2, i % 2
    r = (play(eng, watched, opp, my_ids, opp_ids, seed, mirror=1) if mine == 0
         else play(eng, opp, watched, opp_ids, my_ids, seed, mirror=1))
print(dict(C))
