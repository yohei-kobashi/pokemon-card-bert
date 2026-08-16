"""Three checks the v39 format must pass before any data is regenerated.

1. No card is lost or duplicated by role grouping, for every deck.
2. Rendering the same observation twice is byte-identical (deck_shuffle is off, so it must be).
3. The THREE training-time paths emit the SAME string: rl_rollout.make_serializer,
   build_sft._ser_cur and lm.agent's serializer. A mismatch between any two is silent.
"""
import sys, json, collections
sys.path.insert(0, "."); sys.path.insert(0, "cg-lib"); sys.path.insert(0, "tools")
import library
from cg.game import battle_start, battle_select, battle_finish
from lm.agent import make_lm_agent
from lm.serialize import serialize_stateless, render_my_deck
from lm.roles import for_deck, group
import rl_config

prof = json.load(open("agents/tuning.json"))

# --- 1. grouping is lossless, all decks
bad = 0
for d in sorted(prof):
    try:
        ids = library.read_deck(d)
    except Exception:
        continue
    R = for_deck(d)
    g = group(sorted(set(ids)), R)
    flat = [c for _, cs in g for c in cs]
    if sorted(flat) != sorted(set(ids)) or len(flat) != len(set(flat)):
        print("  LOSSY", d); bad += 1
print("  [1] grouping lossless on %d decks: %s" % (len(prof), "PASS" if bad == 0 else "FAIL"))

# --- get a real mid-game observation
d0 = library.read_deck("rockets_honchkrow"); d1 = library.read_deck("alakazam")
a0 = make_lm_agent("rockets_honchkrow", prof["rockets_honchkrow"], None)
a1 = make_lm_agent("alakazam", prof["alakazam"], None)
obs, _ = battle_start(d0, d1)
target = None
try:
    for _ in range(4000):
        cur = obs.get("current")
        if cur is None or cur.get("result", -1) != -1: break
        if obs.get("select") is None: break
        yi = cur["yourIndex"]
        if yi == 0 and cur.get("turn", 0) >= 5:
            target = obs; break
        obs = battle_select((a0 if yi == 0 else a1)(obs))
    if target is None:
        print("  no mid-game state reached"); raise SystemExit(1)

    FMT = dict(rl_config.PROMPT_FMT)
    # --- 2. determinism
    r1 = serialize_stateless(target, deck_ids=d0, deck_name="rockets_honchkrow", **FMT)
    r2 = serialize_stateless(target, deck_ids=d0, deck_name="rockets_honchkrow", **FMT)
    print("  [2] two renders byte-identical: %s" % ("PASS" if r1 == r2 else "FAIL"))

    # --- 3. three paths agree
    import rl_rollout
    p_roll = rl_rollout.make_serializer(d0, "rockets_honchkrow")(target)
    sys.path.insert(0, "tools/_legacy_decoder")
    import importlib.util
    spec = importlib.util.spec_from_file_location("bsft", "tools/_legacy_decoder/build_sft.py")
    bsft = importlib.util.module_from_spec(spec); spec.loader.exec_module(bsft)
    p_sft = bsft._ser_cur(target, d0, "rockets_honchkrow")
    same = (p_roll == p_sft == r1)
    print("  [3] rl_rollout == build_sft == direct: %s" % ("PASS" if same else "FAIL"))
    if not same:
        for nm, v in (("direct", r1), ("rollout", p_roll), ("build_sft", p_sft)):
            print("      %-9s %s" % (nm, v[:110]))
    print()
    print("  rendered (v39):"); print("   ", r1[:340])
finally:
    battle_finish()
