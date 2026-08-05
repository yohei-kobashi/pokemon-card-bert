#!/usr/bin/env python3
"""How often is an option's EFFECT or COST undecidable from the observation alone?

`prompt-lies-about-retreat-cost` found one instance of a general defect: the prompt renders a
card's PRINTED number while the engine acts on a modified one. That fix (lm/costs.py) covers
retreat, and only the part of it derivable from the visible board. This asks the general
question, against the engine source rather than against card text.

`ToJson.h` exports hp/maxHp/appearThisTurn/energies/tools per Pokemon plus four game flags. The
engine additionally carries, and acts on, unions it never emits (`Card::thisTurn`,
`nextEnemyTurnEndState`, `takeAttackDamagePreTurn`, `PlayerState::thisTurn`,
`State::turnHistories` ...). The instrumented build dumps them; this walks real games and counts
how often a non-zero one lands on a decision where it actually bears on an offered option.

Four classes, and the class decides whether a prompt fix is even possible:

    H  history    written on an EARLIER turn -> unreconstructable from a snapshot; a prompt fix
                  needs carried history.  THIS IS THE NUMBER THE AUDIT EXISTS TO PRODUCE.
    T  this turn  written earlier in the CURRENT turn -> only `appear` + 4 game flags survive
    F  future     written now, bites next turn -> planning only; never scored
    C  continual  recomputed from what is in play -> a card table can derive it TODAY
                  (the class lm/costs.py already handles, for retreat only)

and two severities:

    VALUE     the option is offered and its damage/cost differs from what the prompt implies.
              The model cannot tell.  This is the retreat-cost defect's family.
    LEGALITY  the effect gates the option -- and the engine already applies it by leaving the
              option off the menu.  Harmless when choosing; matters only for planning.

Two ways a hidden field reaches an option, kept apart because conflating them inflates the
answer several-fold:

    UNCONDITIONAL  SetProperty.h:CalcDamage adds `attacker.thisTurn.damageChange` and
                   `target.takeDamageChangeNextEnemyTurn` to EVERY attack's damage and zeroes it
                   for `noDamageAttackNextEnemyTurn`; State.h:retreatCost adds
                   `thisTurn.retreatCostChange`.  Non-zero => relevant, no card lookup.
    CONDITIONAL    read only by a card that names the field.  `takeAttackDamagePreTurn` is
                   non-zero whenever anything hit you last turn, but exactly ONE attack in the
                   1,556-attack database reads it.  Gated on DebugCardDeps.

    python3 tools/audit_hidden_state.py --games 6                    # all decks
    python3 tools/audit_hidden_state.py --decks ns_zoroark --games 30 --examples 10
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

# ---- unconditional: field -> (class, severity, option kind, whose board, active-only) --------
# "self" = the acting player's Pokemon (the attacker / the one retreating);
# "opp"  = the defender.  Damage modifiers on the defender only bite the Pokemon actually being
# hit, so they are restricted to the opponent's ACTIVE; a benched one only matters to a spread
# attack and is counted separately rather than folded into the headline.
CARD_UNCOND = {
    "thisTurn.damageChange":            ("H", "VALUE", "attack", "self", True),
    "thisTurn.damageChangeActive":      ("H", "VALUE", "attack", "self", True),
    "thisTurn.damageChangeMyAttack":    ("H", "VALUE", "attack", "self", True),
    "thisTurn.attackCostChange":        ("H", "VALUE", "attack", "self", True),
    "thisTurn.retreatCostChange":       ("H", "VALUE", "retreat", "self", True),
    "thisTurn.attackCoin":              ("H", "VALUE", "attack", "self", True),
    "thisTurn.attackCoin2":             ("H", "VALUE", "attack", "self", True),
    "thisTurnEnemy.takeDamageChange":   ("H", "VALUE", "attack", "opp", True),
    "takeDamageChangeNextEnemyTurn":    ("H", "VALUE", "attack", "opp", True),
    "noDamageLessEqualAttackNextEnemyTurn":        ("H", "VALUE", "attack", "opp", True),
    "noDamageAndEffectAttackNextEnemyTurn":        ("H", "VALUE", "attack", "opp", True),
    "noDamageAndEffectEnemyAttackNextEnemyTurn":   ("H", "VALUE", "attack", "opp", True),
    "noDamageAttackNextEnemyTurn":                 ("H", "VALUE", "attack", "opp", True),
    "noDamageBasicAttackNextEnemyTurn":            ("H", "VALUE", "attack", "opp", True),
    "noDamageBasicColorAttackNextEnemyTurn":       ("H", "VALUE", "attack", "opp", True),
    "noDamageAbilityAttackNextEnemyTurn":          ("H", "VALUE", "attack", "opp", True),
    "noWeaknessNextEnemyTurn":                     ("H", "VALUE", "attack", "opp", True),
    "noDamageAndEffectEnemyExAttackNextEnemyTurn": ("H", "VALUE", "attack", "opp", True),
    "thisTurn.cannotUseAttackId":       ("H", "LEGALITY", "attack", "self", True),
    "thisTurn.cannotUseAttackId2":      ("H", "LEGALITY", "attack", "self", True),
    "cannotUseAttackIdNonActive":       ("H", "LEGALITY", "attack", "self", True),
    "thisTurn.cannotAttack":            ("H", "LEGALITY", "attack", "self", True),
    "thisTurn.cannotAttackLessEqualEnergy2": ("H", "LEGALITY", "attack", "self", True),
    "thisTurn.cannotRetreat":           ("H", "LEGALITY", "retreat", "self", True),
    "thisTurn.cannotHandAttachEnergy":  ("H", "LEGALITY", "attach", "self", False),
    "damageChangeThisTurn":             ("T", "VALUE", "attack", "self", True),
    "damageChangeExThisTurn":           ("T", "VALUE", "attack", "self", True),
    "koPrizeChange":                    ("T", "VALUE", "attack", "opp", True),
    "koPrizeChangeAlways":              ("T", "VALUE", "attack", "opp", True),
    "koPrizePlus1":                     ("T", "VALUE", "attack", "opp", True),
    "koPrizeZero":                      ("T", "VALUE", "attack", "opp", True),
    "koPrizeDecreaseOnce":              ("T", "VALUE", "attack", "opp", True),
    "evolved":                          ("T", "LEGALITY", "evolve", "self", False),
    "benchToActive":                    ("T", "LEGALITY", "retreat", "self", True),
    "abilityUsed":                      ("T", "LEGALITY", "ability", "self", False),
    "damageChange":                     ("C", "VALUE", "attack", "self", True),
    "damageChangeActive":               ("C", "VALUE", "attack", "self", True),
    "damageChangeEx":                   ("C", "VALUE", "attack", "self", True),
    "damageChangeAbility":              ("C", "VALUE", "attack", "self", True),
    "damageChangeEvolved":              ("C", "VALUE", "attack", "self", True),
    "damageChangeEnemyTakenPrize":      ("C", "VALUE", "attack", "self", True),
    "attackCostDown":                   ("C", "VALUE", "attack", "self", True),
    "attackCostChangeColorless":        ("C", "VALUE", "attack", "self", True),
    "attackCostDownColorlessOwnAttack": ("C", "VALUE", "attack", "self", True),
    "attackEnergyColoressOne":          ("C", "VALUE", "attack", "self", True),
    "attackEnergyPsychicOne":           ("C", "VALUE", "attack", "self", True),
    "doubleGrassEnergy":                ("C", "VALUE", "attack", "self", True),
    "doubleAttack":                     ("C", "VALUE", "attack", "self", True),
    "noDamageCoin":                     ("C", "VALUE", "attack", "self", True),
    "takeDamageChange":                 ("C", "VALUE", "attack", "opp", True),
    "takeEnemyAttackDamageChange":      ("C", "VALUE", "attack", "opp", True),
    "takeEnemyAbilityPokemonAttackDamageChange":    ("C", "VALUE", "attack", "opp", True),
    "takeEnemyFireOrWaterPokemonAttackDamageChange": ("C", "VALUE", "attack", "opp", True),
    "takeEnemy4TypePokemonAttackDamageChange":      ("C", "VALUE", "attack", "opp", True),
    "noDamageGreaterEqual":             ("C", "VALUE", "attack", "opp", True),
    "noDamageEnemyAttack":              ("C", "VALUE", "attack", "opp", True),
    "noEffectEnemyAttack":              ("C", "VALUE", "attack", "opp", True),
    "noDamageEnemyAbilityPokemonAttack": ("C", "VALUE", "attack", "opp", True),
    "noDamageEnemyExAttack":            ("C", "VALUE", "attack", "opp", True),
    "noDamageEnemyBasicExAttack":       ("C", "VALUE", "attack", "opp", True),
    "noDamageAndEffectEnemyTerastalAttack":         ("C", "VALUE", "attack", "opp", True),
    "noDamageAndEffectEnemySpecialEnergyAttack":    ("C", "VALUE", "attack", "opp", True),
    "noDamageCounterEnemyAttackAbility": ("C", "VALUE", "attack", "opp", True),
    "noSpecialCondition":               ("C", "VALUE", "attack", "opp", True),
    "noSleepParalyzeConfuse":           ("C", "VALUE", "attack", "opp", True),
    "noSleep":                          ("C", "VALUE", "attack", "opp", True),
    "noPrizeEx":                        ("C", "VALUE", "attack", "opp", True),
    "basicPrizePlus1":                  ("C", "VALUE", "attack", "opp", True),
    "koByDamageToHand":                 ("C", "VALUE", "attack", "opp", True),
    "weaknessIndex":                    ("C", "VALUE", "attack", "opp", True),
    "typeIndex":                        ("C", "VALUE", "attack", "opp", True),
    "retreatCostChange":                ("C", "VALUE", "retreat", "self", True),
    "noRetreatCost":                    ("C", "VALUE", "retreat", "self", True),
    "noEffectEnemyItem":                ("C", "LEGALITY", "play", "opp", False),
    "noEffectEnemySupporter":           ("C", "LEGALITY", "play", "opp", False),
    "noAbility":                        ("C", "LEGALITY", "ability", "self", False),
    "noEnemyAbility":                   ("C", "LEGALITY", "ability", "self", False),
    "cannotRetreat":                    ("C", "LEGALITY", "retreat", "self", True),
    "cannotAttack":                     ("C", "LEGALITY", "attack", "self", True),
    "canUsePreEvolutionAttack":         ("C", "LEGALITY", "attack", "self", True),
    "canAttackFirst":                   ("C", "LEGALITY", "attack", "self", True),
    # hpChange is already baked into the observation's maxHp by ToJson -> never scored.
}

PLAYER_UNCOND = {
    "thisTurn.metalDamageChange":       ("H", "VALUE", "attack", "self"),
    "playerDamageChange":               ("T", "VALUE", "attack", "self"),
    "playerDamageChangeEx":             ("T", "VALUE", "attack", "self"),
    "playerDamageChangeMyFighting":     ("T", "VALUE", "attack", "self"),
    "takePrizeCountChangeTerastalAttackKoActive": ("T", "VALUE", "attack", "self"),
    "takePrizeCountChangeNAttackKoActive":        ("T", "VALUE", "attack", "self"),
    "poisonDamageCounter":              ("C", "VALUE", "*", "any"),
    "poisonDamageChange":               ("C", "VALUE", "*", "any"),
    "burnDamageChange":                 ("C", "VALUE", "*", "any"),
    "thisTurn.cannotPlayItem":          ("H", "LEGALITY", "play", "self"),
    "thisTurn.cannotPlaySupporter":     ("H", "LEGALITY", "play", "self"),
    "thisTurn.cannotPlayStadium":       ("H", "LEGALITY", "play", "self"),
    "thisTurn.cannotPlaySpecialEnergy": ("H", "LEGALITY", "attach", "self"),
    "thisTurn.cannotEvolve":            ("H", "LEGALITY", "evolve", "self"),
    "thisTurn.cannotRetreatPoison":     ("H", "LEGALITY", "retreat", "self"),
    "thisTurn.cannotAttackLessEqualEnergy2": ("H", "LEGALITY", "attack", "self"),
    "cannotPlayItem":                   ("C", "LEGALITY", "play", "self"),
    "cannotPlayStadium":                ("C", "LEGALITY", "play", "self"),
    "cannotPlayTool":                   ("C", "LEGALITY", "play", "self"),
    "cannotPlayAceSpec":                ("C", "LEGALITY", "play", "self"),
    "cannotPlayAbilityPokemonNotRocket": ("C", "LEGALITY", "play", "self"),
    "cannotTrashToHandAbilityOrTrainers": ("C", "LEGALITY", "play", "self"),
}

# ---- conditional: DebugCardDeps tag -> the hidden field it reads ------------------------------
# scope "game" reads State::turnHistories; "card/self" reads the attacker's own field.
COND = {
    "H:koPreEnemyTurn":            ("game", "h1.ko", "VALUE"),
    "H:koPreEnemyTurnTR":          ("game", "h1.koTeamRocket", "VALUE"),
    "H:koAtkDmgPreEnemyTurn":      ("game", "h1.koAttackDamage", "VALUE"),
    "H:koAtkDmgEthan":             ("game", "h1.koAttackDamageEthan", "VALUE"),
    "H:koAtkDmgHop":               ("game", "h1.koAttackDamageHop", "VALUE"),
    "H:dmgFromPreTurnPrizeCount":  ("game", "h1.takePrizeCount", "VALUE"),
    "H:targetPreTurnAttacker":     ("game", "h1.turnAttackId", "VALUE"),
    "H:sameAttackPreMyTurn":       ("game", "h2.turnAttackId", "VALUE"),
    "H:cantRepeatAttack":          ("game", "h2.turnAttackId", "LEGALITY"),
    "H:dmgFromTakeDamagePreTurn":  ("self", "takeAttackDamagePreTurn", "VALUE"),
    "T:coinHeadCount":             ("game", "coinHeadCount", "VALUE"),
    "T:noSameNameSkillThisTurn":   ("card", "abilityUsed", "LEGALITY"),
    "T:attachActive":              ("game", "-", "VALUE"),
}


def menu(obs):
    """-> (kinds present, attack ids offered)."""
    from lm.actions import encode_option
    kinds, aids = set(), set()
    for o in (obs.get("select") or {}).get("option") or []:
        t = encode_option(o, obs)
        kinds.add(t.split(":")[0])
        if t.startswith("attack:"):
            try:
                aids.add(int(t.split(":")[1]))
            except ValueError:
                pass
    return kinds, aids


def scan(hid, obs, yi, deps):
    """-> list of (class, severity, scope, field) that BEAR on an option actually offered."""
    kinds, aids = menu(obs)
    hits = []

    # index the hidden state for the conditional lookups
    game = hid.get("game") or {}
    self_active = {}
    for c in hid.get("cards") or []:
        if c.get("pi") == yi and c.get("z") == "A":
            for cls in ("H", "T", "C"):
                self_active.update(c.get(cls) or {})

    for c in hid.get("cards") or []:
        side = "self" if c.get("pi") == yi else "opp"
        active = c.get("z") == "A"
        for cls in ("H", "T", "C"):
            for field, val in (c.get(cls) or {}).items():
                spec = CARD_UNCOND.get(field)
                if not spec:
                    continue
                fcls, sev, kind, want, act_only = spec
                if want != side or (act_only and not active):
                    continue
                if kind not in kinds:
                    continue
                hits.append((cls, sev, "card/" + side + ("" if active else "/bench"), field))
    for p in hid.get("players") or []:
        side = "self" if p.get("pi") == yi else "opp"
        for cls in ("H", "T", "C"):
            for field in (p.get(cls) or {}):
                spec = PLAYER_UNCOND.get(field)
                if not spec:
                    continue
                fcls, sev, kind, want = spec
                if want != "any" and want != side:
                    continue
                if kind != "*" and kind not in kinds:
                    continue
                hits.append((cls, sev, "player/" + side, field))

    # conditional: only when an OFFERED attack names the field
    for aid in aids:
        for tag in deps.get(str(aid), ()):
            scope, field, sev = COND.get(tag, (None, None, None))
            if scope is None:
                continue
            cls = tag.split(":")[0]
            if scope == "game" and game.get(field):
                hits.append((cls, sev, "game/cond", "%s <- a%d" % (field, aid)))
            elif scope == "self" and self_active.get(field):
                hits.append((cls, sev, "card/self/cond", "%s <- a%d" % (field, aid)))
    return hits


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--decks", default="")
    ap.add_argument("--games", type=int, default=6)
    ap.add_argument("--seed-base", type=int, default=500000)
    ap.add_argument("--so", default=HIDDEN_SO)
    ap.add_argument("--examples", type=int, default=8)
    ap.add_argument("--min-options", type=int, default=2,
                    help="ignore forced decisions -- a 1-option menu is not a choice")
    ap.add_argument("--shard", default="", help="i/n: take every n-th deck")
    ap.add_argument("--opp", default="mirror",
                    help="'mirror' (same deck both seats), 'live' (rotate the LIVE_META top 10) "
                         "or a comma list. Mirror cannot show an effect imposed by a card the "
                         "deck does not itself run, so 'live' is the faithful field.")
    ap.add_argument("--out", default="", help="write counts as JSON")
    a = ap.parse_args()

    import library
    from lm.agent import make_lm_agent
    from tools.mirror_env import MirrorEngine

    eng = MirrorEngine(a.so)
    if not eng.has_hidden:
        sys.exit("%s has no DebugHiddenState -- rebuild:\n  python3 tools/build_engine_mirror.py "
                 "--out data/kaggle_engine_ext/libcg_hidden.so" % a.so)
    deps = eng.card_deps()["attacks"]
    tuning = json.load(open(os.path.join(ROOT, "agents", "tuning.json")))
    decks = [d.strip() for d in a.decks.split(",") if d.strip()] or sorted(library.list_decks())
    if a.shard:
        i, n = (int(x) for x in a.shard.split("/"))
        decks = decks[i::n]

    if a.opp == "live":
        from tools.rl_config import LIVE_META
        opps = [d for d, _ in sorted(LIVE_META.items(), key=lambda kv: -kv[1])[:10]
                if d in set(library.list_decks())]
    elif a.opp == "mirror":
        opps = None
    else:
        opps = [d.strip() for d in a.opp.split(",") if d.strip()]

    def load(name):
        return [int(x) for x in open(library.deck_path(name)) if x.strip()]

    tot = collections.Counter()
    by_field = collections.Counter()
    per_deck = {}
    examples = []
    for di, deck in enumerate(decks):
        ids = load(deck)
        agent = make_lm_agent(ids, tuning.get(deck, {}), model=None)
        st = collections.Counter()
        for g in range(a.games):
            if opps is None:
                oname, oids = deck, list(ids)
            else:
                oname = opps[g % len(opps)]
                oids = load(oname)
            oagent = make_lm_agent(oids, tuning.get(oname, {}), model=None)
            obs = eng.start(ids, oids, a.seed_base + di * 1000 + g, mirror=1 if opps is None else 0)
            if obs is None:
                continue
            try:
                for _ in range(4000):
                    cur = obs.get("current") or {}
                    if cur.get("result", -1) != -1 or not obs.get("select"):
                        break
                    opts = (obs.get("select") or {}).get("option") or []
                    yi = cur.get("yourIndex", 0)
                    # Score OUR seat only, so a rate is attributable to `deck` and not to
                    # whatever the rotating opponent happened to be.
                    if yi == 0 and len(opts) >= a.min_options:
                        hits = scan(eng.hidden_state(), obs, yi, deps)
                        st["decisions"] += 1
                        seen = set()
                        for cls, sev, scope, field in hits:
                            by_field[(cls, sev, scope, field.split(" <- ")[0])] += 1
                            seen.add((cls, sev))
                        for cls, sev in seen:
                            st["%s_%s" % (cls, sev)] += 1
                        if ("H", "VALUE") in seen:
                            st["H_VALUE"] += 0    # already counted; keep key present
                            if len(examples) < a.examples:
                                examples.append((deck, cur.get("turn"),
                                                 sorted(menu(obs)[0]),
                                                 [h for h in hits if h[0] == "H" and h[1] == "VALUE"]))
                        if any(s == "VALUE" and c in ("H", "T") for c, s in seen):
                            st["HT_VALUE"] += 1
                        if any(s == "VALUE" for c, s in seen):
                            st["ANY_VALUE"] += 1
                    obs = eng.select((agent if yi == 0 else oagent)(obs))
            except Exception:
                pass
            finally:
                eng.finish()
        per_deck[deck] = dict(st)
        tot.update(st)
        n = st["decisions"] or 1
        print("  %-24s %6d dec   H-val %5.1f%%  T-val %5.1f%%  C-val %5.1f%%"
              % (deck, st["decisions"], 100 * st["H_VALUE"] / n,
                 100 * st["T_VALUE"] / n, 100 * st["C_VALUE"] / n), flush=True)

    n = tot["decisions"] or 1
    print("\n%d decisions (>=%d options) over %d decks x %d games\n"
          % (tot["decisions"], a.min_options, len(decks), a.games))
    print("%-46s%10s%8s" % ("class / severity", "decisions", "share"))
    for cls, note in (("H", "previous turn -- prompt CANNOT say"),
                      ("T", "earlier this turn -- prompt cannot say"),
                      ("C", "derivable from the visible board")):
        for sev in ("VALUE", "LEGALITY"):
            k = "%s_%s" % (cls, sev)
            print("%-46s%10d%7.1f%%%s" % (k, tot[k], 100 * tot[k] / n,
                                          "   " + note if sev == "VALUE" else ""))
    print("%-46s%10d%7.1f%%" % ("H+T VALUE (not in the snapshot at all)", tot["HT_VALUE"],
                                100 * tot["HT_VALUE"] / n))
    print("%-46s%10d%7.1f%%" % ("ANY VALUE", tot["ANY_VALUE"], 100 * tot["ANY_VALUE"] / n))

    print("\nfields that bear on an offered option, most frequent first:")
    print("%-4s%-10s%-18s%-40s%9s%8s" % ("cls", "severity", "scope", "field", "hits", "share"))
    for (cls, sev, scope, field), c in by_field.most_common(45):
        print("%-4s%-10s%-18s%-40s%9d%7.1f%%" % (cls, sev, scope, field, c, 100 * c / n))

    if examples:
        print("\nH/VALUE examples (previous-turn effect, option offered, prompt cannot say):")
        for deck, turn, kinds, hits in examples:
            print("  %-20s T%-3s menu=%s" % (deck, turn, ",".join(kinds)))
            for cls, sev, scope, field in hits:
                print("      %-18s %s" % (scope, field))

    if a.out:
        json.dump({"per_deck": per_deck, "total": dict(tot),
                   "by_field": {"|".join(k): v for k, v in by_field.items()}},
                  open(a.out, "w"), indent=1)
        print("\nwrote %s" % a.out)


if __name__ == "__main__":
    main()
