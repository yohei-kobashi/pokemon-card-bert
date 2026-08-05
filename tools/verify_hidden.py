#!/usr/bin/env python3
"""Does lm/hidden.py's decode of the observation blob equal the engine's own state?

lm/hidden.py reads `obs["search_begin_input"]` -- the raw POD dump `State::serialize` produces --
using BAKED byte offsets and a hand-derived bitfield layout. Both are exactly the kind of thing
that is silently wrong: a mis-assigned bit yields a plausible number, not a crash.

So it is checked against ground truth. The instrumented build exports `DebugHiddenState`, which
reads the SAME fields through the C++ struct; every non-zero field either side reports must match.
Run this before trusting the decode, and again after any engine refetch.

    python3 tools/verify_hidden.py --games 4              # every deck
    python3 tools/verify_hidden.py --decks slowking --games 20 --verbose
"""

import argparse
import collections
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "cg-lib")):
    if p not in sys.path:
        sys.path.insert(0, p)

HIDDEN_SO = os.path.join(ROOT, "data", "kaggle_engine_ext", "libcg_hidden.so")

# DebugHiddenState key -> where the same value lives in lm.hidden's decode of one card.
# Keys it does not list are not cross-checked (the C++ side simply does not emit them).
CARD_MAP = {
    "thisTurn.": ("thisTurn", None), "nextTurn.": ("nextTurn", None),
    "thisTurnEnemy.": ("thisTurnEnemy", None),
}
CARD_DIRECT = {
    "takeAttackDamagePreTurn": ("takeAttackDamagePreTurn",),
    "cannotUseAttackIdNonActive": ("cannotUseAttackIdNonActive",),
    "noDamageAndEffectEnemyExAttackNextEnemyTurn":
        ("noDamageAndEffectEnemyExAttackNextEnemyTurn",),
}
NETE = {"takeDamageChangeNextEnemyTurn", "noDamageLessEqualAttackNextEnemyTurn",
        "noDamageAndEffectAttackNextEnemyTurn", "noDamageAndEffectEnemyAttackNextEnemyTurn",
        "noDamageAttackNextEnemyTurn", "noDamageBasicAttackNextEnemyTurn",
        "noDamageBasicColorAttackNextEnemyTurn", "noDamageAbilityAttackNextEnemyTurn",
        "noWeaknessNextEnemyTurn"}
TURNSTATE = {"damageChangeThisTurn", "damageChangeExThisTurn", "koPrizeChangeAlways",
             "koPrizeChange", "evolved", "benchToActive", "koPrizePlus1", "koPrizeDecreaseOnce",
             "koPrizeZero"}
SKIP_CARD = {"takeAttackDamageThisTurn", "abilityUsed"}     # not decoded (LEGALITY-only)
PLAYER_TT = {"metalDamageChange", "cannotAttackLessEqualEnergy2", "cannotPlayItem",
             "cannotPlaySupporter", "cannotPlayStadium", "cannotPlaySpecialEnergy",
             "cannotEvolve", "cannotRetreatPoison"}
PLAYER_TS = {"playerDamageChange", "playerDamageChangeEx", "playerDamageChangeMyFighting"}
PLAYER_CONT = {"poisonDamageChange", "burnDamageChange", "poisonDamageCounter"}
SKIP_PLAYER = {"takePrizeCountChangeTerastalAttackKoActive",
               "takePrizeCountChangeNAttackKoActive", "cannotPlayTool", "cannotPlayAceSpec",
               "cannotPlayAbilityPokemonNotRocket", "cannotTrashToHandAbilityOrTrainers",
               "cannotPlayStadium", "cannotPlayItem"}   # player continual bits, not decoded


def card_lookup(dec, field):
    for prefix, (group, _) in CARD_MAP.items():
        if field.startswith(prefix):
            return dec[group].get(field[len(prefix):], "MISSING")
    if field in CARD_DIRECT:
        return dec.get(field, "MISSING")
    if field in NETE:
        return dec["nextEnemyTurnEndState"].get(field, "MISSING")
    if field in TURNSTATE:
        return dec["turnState"].get(field, "MISSING")
    return dec["continual"].get(field, "MISSING")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--decks", default="")
    ap.add_argument("--games", type=int, default=4)
    ap.add_argument("--seed-base", type=int, default=310000)
    ap.add_argument("--opp", default="live")
    ap.add_argument("--shard", default="")
    ap.add_argument("--so", default=HIDDEN_SO)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    import library
    from lm import hidden
    from lm.agent import make_lm_agent
    from tools.mirror_env import MirrorEngine

    eng = MirrorEngine(a.so)
    if not eng.has_hidden:
        sys.exit("need the instrumented build (libcg_hidden.so)")
    tuning = json.load(open(os.path.join(ROOT, "agents", "tuning.json")))
    decks = [d.strip() for d in a.decks.split(",") if d.strip()] or sorted(library.list_decks())
    if a.shard:
        i, n = (int(x) for x in a.shard.split("/"))
        decks = decks[i::n]
    if a.opp == "live":
        from tools.rl_config import LIVE_META
        opps = [d for d, _ in sorted(LIVE_META.items(), key=lambda kv: -kv[1])[:10]
                if d in set(library.list_decks())]
    else:
        opps = None

    def load(n):
        return [int(x) for x in open(library.deck_path(n)) if x.strip()]

    st = collections.Counter()
    bad = collections.Counter()
    examples = []
    for di, deck in enumerate(decks):
        ids = load(deck)
        agent = make_lm_agent(ids, tuning.get(deck, {}), model=None)
        for g in range(a.games):
            oname = deck if opps is None else opps[g % len(opps)]
            oids = list(ids) if opps is None else load(oname)
            oagent = make_lm_agent(oids, tuning.get(oname, {}), model=None)
            obs = eng.start(ids, oids, a.seed_base + di * 1000 + g,
                            mirror=1 if opps is None else 0)
            if obs is None:
                continue
            try:
                for _ in range(4000):
                    cur = obs.get("current") or {}
                    if cur.get("result", -1) != -1 or not obs.get("select"):
                        break
                    yi = cur.get("yourIndex", 0)
                    truth = eng.hidden_state()
                    dec = hidden.read(obs)
                    st["decisions"] += 1
                    if dec is None:
                        st["decode_none"] += 1
                    else:
                        st["decoded"] += 1
                        by_serial = {s: (pi, z, i)
                                     for pi, z, i, s, m in hidden.in_play_serials(obs)}
                        slot2serial = {(pi, z, i): s for s, (pi, z, i) in by_serial.items()}
                        for c in truth.get("cards") or []:
                            key = (c["pi"], c["z"], c["i"])
                            serial = slot2serial.get(key)
                            if serial is None:
                                st["slot_unmatched"] += 1
                                continue
                            d = dec["cards"].get(serial)
                            for grp in ("H", "T", "C"):
                                for field, val in (c.get(grp) or {}).items():
                                    if field in SKIP_CARD:
                                        continue
                                    st["checked"] += 1
                                    got = card_lookup(d, field)
                                    if got != val:
                                        bad[("card", field)] += 1
                                        if len(examples) < 12:
                                            examples.append((deck, field, val, got))
                        for p in truth.get("players") or []:
                            d = dec["players"][p["pi"]]
                            for grp in ("H", "T", "C"):
                                for field, val in (p.get(grp) or {}).items():
                                    base = field.split(".")[-1]
                                    if base in SKIP_PLAYER and not field.startswith("thisTurn."):
                                        continue
                                    if field.startswith("thisTurn."):
                                        got = d["thisTurn"].get(base, "MISSING")
                                    elif base in PLAYER_TS:
                                        got = d["turnState"].get(base, "MISSING")
                                    elif base == "poisonDamageCounter":
                                        got = d["poisonDamageCounter"]
                                    elif base in PLAYER_CONT:
                                        got = d["continual"].get(base, "MISSING")
                                    else:
                                        continue
                                    st["checked"] += 1
                                    if got != val:
                                        bad[("player", field)] += 1
                                        if len(examples) < 12:
                                            examples.append((deck, "player." + field, val, got))
                        for k in (1, 2):
                            for short, long in (("ko", "ko"), ("koTeamRocket", "koTeamRocket"),
                                                ("koAttackDamage", "koAttackDamage"),
                                                ("koAttackDamageEthan", "koAttackDamageEthan"),
                                                ("koAttackDamageHop", "koAttackDamageHop"),
                                                ("turnAttackId", "turnAttackId"),
                                                ("takePrizeCount", "takePrizeCount")):
                                val = (truth.get("game") or {}).get("h%d.%s" % (k, short), 0)
                                got = dec["history"][k][long]
                                st["checked"] += 1
                                if got != val:
                                    bad[("game", "h%d.%s" % (k, short))] += 1
                                    if len(examples) < 12:
                                        examples.append((deck, "h%d.%s" % (k, short), val, got))
                    obs = eng.select((agent if yi == 0 else oagent)(obs))
            except Exception:
                pass
            finally:
                eng.finish()
        if a.verbose:
            print("  %-24s checked %d, mismatches %d" % (deck, st["checked"], sum(bad.values())),
                  flush=True)

    print("\n%d decisions | decoded %d | blob missing/rejected %d | slot unmatched %d"
          % (st["decisions"], st["decoded"], st["decode_none"], st["slot_unmatched"]))
    print("%d field comparisons, %d mismatches" % (st["checked"], sum(bad.values())))
    if bad:
        print("\nMISMATCHED FIELDS (bit layout is wrong for these):")
        for (scope, field), c in bad.most_common(40):
            print("  %-8s %-52s %d" % (scope, field, c))
        print("\nexamples (engine -> decoded):")
        for deck, field, val, got in examples:
            print("  %-20s %-52s %s -> %s" % (deck, field, val, got))
    return 1 if bad or st["decode_none"] else 0


if __name__ == "__main__":
    sys.exit(main())
