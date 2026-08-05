"""The engine's hidden effect state, read from the blob the OFFICIAL library already hands us.

`ToJson.h` exports hp / maxHp / appearThisTurn / energies / energyCards / tools per Pokemon and
four game flags. The engine also carries unions it acts on and never emits -- damage modifiers,
damage nullification, live retreat and attack costs, `thisTurn` / `nextEnemyTurnEndState`
history. `tools/audit_hidden_state.py` measured them at 17.6% of decisions
([[hidden-effect-state-audit]]).

There is no need to model any of it. `Api.h:ApiGetBattleData` calls `State::serialize`, which is

    b.set(&turn, &options);

-- a raw copy of the POD region, and that region spans `players` and `allCard[128]`. So the
base64 already in `obs["search_begin_input"]` carries EVERY hidden union, produced by the
official libcg.so, needing no extra engine call and no card table. This module decodes it.

Two things make that safe to depend on:

  * every value is CROSS-CHECKED against the JSON before use (`verify`): the decoded cardId,
    area and damage of each in-play Pokemon must match what `ToJson` said. A layout mismatch --
    a differently-built libcg.so, a schema change -- fails that check on the first decision, and
    `read()` returns None rather than a plausible-looking wrong number.
  * the offsets are BAKED, not derived. tools/build_engine_mirror.py's DebugLayout emits them
    and tools/verify_hidden.py asserts the decode equals DebugHiddenState field by field.

Bitfield layout is the GCC x86-64 rule (little-endian, allocated from the LSB of the storage
unit in declaration order). Hand-deriving it is exactly the kind of thing that is silently wrong,
which is why verify_hidden.py compares against the engine rather than against reasoning.
"""

import base64
import struct

# ---- baked layout (tools/build_engine_mirror.py --out ... ; DebugLayout) ----------------------
_S_TURN = 0
_S_TURN_ACTION = 4
_S_TURN_HISTORIES = 220
_S_PLAYERS = 480
_S_ALLCARD = 1672
_S_POD = 22152
_SZ_CARD = 160
_SZ_PLAYER = 592
_SZ_TURNHIST = 12

_C_DAMAGE = 16
_C_THIS_TURN = 20
_C_NEXT_TURN = 36
_C_THIS_TURN_ENEMY = 52
_C_TAKE_DMG_PRE = 64
_C_PLAYER_INDEX = 68
_C_AREA = 69
_C_NO_ATTACK_NONACTIVE = 72
_C_NETE_BF = 96
_C_NETE = 100
_C_TURN_STATE = 104
_C_CONTINUAL = 120

_P_PLAYER_INDEX = 556
_P_THIS_TURN = 560
_P_ACTIVE_STATE = 568
_P_CONTINUAL = 576
_P_TURN_STATE = 584

_B64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
_VAL = {c: i for i, c in enumerate(_B64)}


def decode_blob(s):
    """`obs["search_begin_input"]` -> raw bytes.

    Plain base64 over `_B64`, except that runs of 'A' (the zero sextet, and the state is mostly
    zeros) are run-length coded by BinaryWriter::addA: 'A'+1 char, '-'+2, '*'+3, little-end
    base-64 digits. Mirrors BinaryReader::fromBase64.
    """
    out = []
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if c == "A":
            out.append("A" * _VAL[s[i + 1]])
            i += 2
        elif c == "-":
            out.append("A" * (_VAL[s[i + 1]] + 64 * _VAL[s[i + 2]]))
            i += 3
        elif c == "*":
            out.append("A" * (_VAL[s[i + 1]] + 64 * _VAL[s[i + 2]] + 4096 * _VAL[s[i + 3]]))
            i += 4
        else:
            out.append(c)
            i += 1
    t = "".join(out)
    return base64.b64decode(t + "=" * (-len(t) % 4))


def _bits(buf, off, names):
    """`names` is a list of (byte offset from `off`, bit index, name) in declaration order."""
    return {nm: (buf[off + b] >> k) & 1 for b, k, nm in names}


# CardNextTurnState: 5 shorts, 2 signed chars, then 6 bools packed into the next byte.
_NEXT_TURN_BITS = [(12, 0, "cannotRetreat"), (12, 1, "cannotHandAttachEnergy"),
                   (12, 2, "cannotAttack"), (12, 3, "cannotAttackLessEqualEnergy2"),
                   (12, 4, "attackCoin"), (12, 5, "attackCoin2")]

# nextEnemyTurnEndState: short + unsigned char + 7 bools.
_NETE_BITS = [(3, 0, "noDamageAndEffectAttackNextEnemyTurn"),
              (3, 1, "noDamageAndEffectEnemyAttackNextEnemyTurn"),
              (3, 2, "noDamageAttackNextEnemyTurn"),
              (3, 3, "noDamageBasicAttackNextEnemyTurn"),
              (3, 4, "noDamageBasicColorAttackNextEnemyTurn"),
              (3, 5, "noDamageAbilityAttackNextEnemyTurn"),
              (3, 6, "noWeaknessNextEnemyTurn")]

# Card::continualState -- 13 shorts, 6 signed chars, then 40 bools in declaration order.
_CONT_SHORTS = ["hpChange", "damageChange", "damageChangeActive", "damageChangeEx",
                "damageChangeAbility", "damageChangeEvolved", "damageChangeEnemyTakenPrize",
                "takeDamageChange", "takeEnemyAttackDamageChange",
                "takeEnemyAbilityPokemonAttackDamageChange",
                "takeEnemyFireOrWaterPokemonAttackDamageChange",
                "takeEnemy4TypePokemonAttackDamageChange", "noDamageGreaterEqual"]
_CONT_CHARS = ["retreatCostChange", "attackCostChangeColorless", "attackCostDown",
               "attackCostDownColorlessOwnAttack", "typeIndex", "weaknessIndex"]
_CONT_FLAGS = ["noAbility", "noKoMeAbility", "noDamageEnemyAbilityPokemonAttack",
               "noDamageEnemyExAttack", "noDamageEnemyBasicExAttack",
               "noDamageAndEffectEnemyTerastalAttack",
               "noDamageAndEffectEnemySpecialEnergyAttack", "noDamageEnemyAttack",
               "noEffectEnemyAttack", "noEffectEnemyItem", "noEffectEnemySupporter",
               "noDamageCounterEnemyAttackAbility", "noEnemyAbility", "noSpecialCondition",
               "noSleepParalyzeConfuse", "noSleep", "noRetreatCost", "noPrizeEx",
               "notRecoverConfuseEvolve", "canUsePreEvolutionAttack", "canEvolveAppearTurn",
               "canEvolveGrassAppearTurn", "canAttackFirst", "cannotRetreat", "cannotAttack",
               "cannotToHand", "cannotMoveDamageCounter", "attackEnergyColoressOne",
               "attackEnergyPsychicOne", "doubleGrassEnergy", "noDamageCoin", "koByDamageToHand",
               "basicPrizePlus1", "doubleAttack", "tool2", "tool4", "technicalMachine",
               "specialFlagTool", "rainbowDna", "canPlay"]
_CONT_FLAG_BASE = 32          # 13*2 + 6 = 32

# Card::turnState -- 2 shorts, 3 chars, then 17 bools.
_TURNSTATE_FLAGS = ["appear", "evolved", "benchToActive", "ko", "koAttackDamage",
                    "koEnemyAttackDamage", "koEnemyAttackDamageActive", "koEnemyExAttackDamage",
                    "koEnemyTerastalAttackDamage", "koEnemyNAttackDamage", "koFull",
                    "koPrizePlus1", "koPrizeDecreaseOnce", "koPrizeZero",
                    "koNoDamageAndEffectAttackNextEnemyTurn"]
_TURNSTATE_FLAG_BASE = 7      # 2*2 + 3 = 7

_PLAYER_TT_BITS = [(2, 0, "cannotAttackLessEqualEnergy2"), (2, 1, "cannotPlayItem"),
                   (2, 2, "cannotPlaySupporter"), (2, 3, "cannotPlayStadium"),
                   (2, 4, "cannotPlaySpecialEnergy"), (2, 5, "cannotEvolve"),
                   (2, 6, "cannotRetreatPoison")]


def _flags(buf, off, base, names):
    out = {}
    for i, nm in enumerate(names):
        out[nm] = (buf[off + base + (i >> 3)] >> (i & 7)) & 1
    return out


def _card(buf, serial):
    o = _S_ALLCARD + serial * _SZ_CARD
    if o + _SZ_CARD > len(buf):
        return None
    i32 = lambda p: struct.unpack_from("<i", buf, o + p)[0]          # noqa: E731
    i16 = lambda p: struct.unpack_from("<h", buf, o + p)[0]          # noqa: E731
    i8 = lambda p: struct.unpack_from("<b", buf, o + p)[0]           # noqa: E731

    def next_turn(p):
        d = {"cannotUseAttackId": i16(p), "cannotUseAttackId2": i16(p + 2),
             "damageChange": i16(p + 4), "damageChangeActive": i16(p + 6),
             "damageChangeMyAttack": i16(p + 8), "attackCostChange": i8(p + 10),
             "retreatCostChange": i8(p + 11)}
        d.update(_bits(buf, o + p, _NEXT_TURN_BITS))
        return d

    cont = {nm: i16(_C_CONTINUAL + 2 * i) for i, nm in enumerate(_CONT_SHORTS)}
    cont.update({nm: i8(_C_CONTINUAL + 26 + i) for i, nm in enumerate(_CONT_CHARS)})
    cont.update(_flags(buf, o + _C_CONTINUAL, _CONT_FLAG_BASE, _CONT_FLAGS))

    ts = {"damageChangeThisTurn": i16(_C_TURN_STATE), "damageChangeExThisTurn": i16(_C_TURN_STATE + 2),
          "koPrizeChangeAlways": i8(_C_TURN_STATE + 5), "koPrizeChange": i8(_C_TURN_STATE + 6)}
    ts.update(_flags(buf, o + _C_TURN_STATE, _TURNSTATE_FLAG_BASE, _TURNSTATE_FLAGS))

    nete = {"takeDamageChangeNextEnemyTurn": i16(_C_NETE),
            "noDamageLessEqualAttackNextEnemyTurn": buf[o + _C_NETE + 2]}
    nete.update(_bits(buf, o + _C_NETE, _NETE_BITS))

    return {
        "cardId": i32(0), "damage": i32(_C_DAMAGE),
        "playerIndex": i8(_C_PLAYER_INDEX), "area": buf[o + _C_AREA],
        "takeAttackDamagePreTurn": i32(_C_TAKE_DMG_PRE),
        "cannotUseAttackIdNonActive": i16(_C_NO_ATTACK_NONACTIVE),
        "thisTurn": next_turn(_C_THIS_TURN), "nextTurn": next_turn(_C_NEXT_TURN),
        "thisTurnEnemy": {"takeDamageChange": i16(_C_THIS_TURN_ENEMY)},
        "nextEnemyTurnEndState": nete,
        "noDamageAndEffectEnemyExAttackNextEnemyTurn": buf[o + _C_NETE_BF] & 1,
        "turnState": ts, "continual": cont,
    }


def _player(buf, pi):
    o = _S_PLAYERS + pi * _SZ_PLAYER
    i16 = lambda p: struct.unpack_from("<h", buf, o + p)[0]          # noqa: E731
    i8 = lambda p: struct.unpack_from("<b", buf, o + p)[0]           # noqa: E731
    tt = {"metalDamageChange": i16(_P_THIS_TURN)}
    tt.update(_bits(buf, o + _P_THIS_TURN, _PLAYER_TT_BITS))
    return {
        "playerIndex": i8(_P_PLAYER_INDEX),
        "thisTurn": tt,
        "poisonDamageCounter": i8(_P_ACTIVE_STATE),
        "continual": {"poisonDamageChange": i16(_P_CONTINUAL),
                      "burnDamageChange": i16(_P_CONTINUAL + 2)},
        "turnState": {"playerDamageChange": i16(_P_TURN_STATE),
                      "playerDamageChangeEx": i16(_P_TURN_STATE + 2),
                      "playerDamageChangeMyFighting": i16(_P_TURN_STATE + 4)},
    }


def _turn_history(buf, k):
    o = _S_TURN_HISTORIES + k * _SZ_TURNHIST
    return {"ko": buf[o], "koTeamRocket": buf[o + 1], "koAttackDamage": buf[o + 2],
            "koAttackDamageEthan": buf[o + 3], "koAttackDamageHop": buf[o + 4],
            "turnAttackCard": buf[o + 5],
            "takePrizeCount": struct.unpack_from("<b", buf, o + 6)[0],
            "turnAttackId": struct.unpack_from("<i", buf, o + 8)[0]}


def in_play_serials(obs):
    """[(playerIndex, "A"|"B", index, serial, json_pokemon)] for every Pokemon on either board."""
    out = []
    cur = obs.get("current") or {}
    for pi, pl in enumerate(cur.get("players") or []):
        for z, key in (("A", "active"), ("B", "bench")):
            for i, m in enumerate(pl.get(key) or []):
                if m and m.get("serial") is not None:
                    out.append((pi, z, i, m["serial"], m))
    return out


def read(obs):
    """Decoded hidden state, or None if the blob is missing or fails `verify`.

    Returning None -- rather than a best effort -- is deliberate: a wrong damage modifier in the
    prompt is worse than none, and a layout mismatch is exactly the failure that would produce
    plausible wrong numbers.
    """
    blob = obs.get("search_begin_input")
    if not blob:
        return None
    try:
        buf = decode_blob(blob)
    except Exception:
        return None
    if len(buf) < _S_POD:
        return None
    got = {"turn": struct.unpack_from("<i", buf, _S_TURN)[0],
           "cards": {}, "players": [_player(buf, 0), _player(buf, 1)],
           "history": [_turn_history(buf, k) for k in range(3)]}
    for pi, z, i, serial, m in in_play_serials(obs):
        c = _card(buf, serial)
        if c is None:
            return None
        got["cards"][serial] = c
    return got if verify(obs, got) else None


def verify(obs, got):
    """Does the decode agree with the JSON on the fields BOTH carry?

    cardId, board side and damage are emitted by ToJson and sit at three different offsets inside
    Card, so agreeing on all of them across every in-play Pokemon pins the layout. This is the
    guard that makes baked offsets safe to ship.
    """
    cur = obs.get("current") or {}
    if got.get("turn") != cur.get("turn"):
        return False
    n = 0
    for pi, z, i, serial, m in in_play_serials(obs):
        c = got["cards"].get(serial)
        if c is None or c["cardId"] != m.get("id") or c["playerIndex"] != pi:
            return False
        if m.get("maxHp") is not None and m.get("hp") is not None:
            if c["damage"] != m["maxHp"] - m["hp"]:
                return False
        n += 1
    return n > 0


# ------------------------------------------------------------------------------------------
# Derived facts: what the prompt should actually SAY.
#
# The raw unions are not renderable -- 40 flags per Pokemon is nothing like "a reasonable token
# cost". But `SetProperty.h:CalcDamage` collapses them into three quantities per (attacker,
# target) pair, and every input it needs is either visible or decoded here:
#
#     damage = base + ATTACKER_DELTA           (all evaluable)
#              [x2 weakness] [-30 resistance]  (weakness itself is overridable -- weaknessIndex,
#                                               noWeaknessNextEnemyTurn)
#              + TARGET_DELTA                  (all evaluable)
#              [= 0 if a nullification flag matches the attacker's properties]
#
# So one signed number per side plus an immunity flag reproduces the whole hidden damage model
# for the decision actually in front of the model. base damage stays where it already is.
# ------------------------------------------------------------------------------------------

_COLORLESS, _GRASS, _FIRE, _WATER, _LIGHTNING = 0, 1 << 0, 1 << 1, 1 << 2, 1 << 3
_FIGHTING = 1 << 5
_METAL = 1 << 7
_ENERGY_TYPES = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 511, 16 | 64]   # EnergyTypes[], as masks


def _master(cid):
    from lm import vocab
    return vocab._CARDS.get(cid)


def _type_mask(cid, cont):
    m = _master(cid)
    t = 0 if m is None else (1 << (m.energyType - 1) if m.energyType else 0)
    if m is not None and m.energyType == 0:
        t = 0
    ti = cont.get("typeIndex") or 0
    if ti > 0:
        t |= _ENERGY_TYPES[ti]
    return t


def _has_ability(cid, cont):
    m = _master(cid)
    if m is None or not (getattr(m, "skills", None) or []):
        return False
    return not cont.get("noAbility")


def _is_ex(cid):
    m = _master(cid)
    return bool(m is not None and (m.ex or m.megaEx))


def _is_basic(cid):
    m = _master(cid)
    return bool(m is not None and m.basic)


def attack_facts(obs, dec, attacker_pi, attacker_serial, target_serial, target_area="active"):
    """`{"atk", "tgt", "zero", "wk", "nofx", "floor", "cap"}` for one attacker->target pair.

    `atk` is added BEFORE weakness and `tgt` after, which is the order CalcDamage uses, so the
    two cannot be summed into one number. `zero` means the target is immune to THIS attacker --
    the ex / Basic / ability-holder / Terastal / special-energy conditions are all resolved
    against the attacker actually in play. `nofx` is Mist / Rock Fighting Energy: the damage
    still lands, the attack's EFFECT does not.

    Several blocks apply only when the two are on OPPOSITE sides (CalcDamage guards them with
    `attacker.playerIndex != target.playerIndex`); self-targeting attacks exist, so that guard
    is carried here rather than assumed away. tools/verify_damage.py caught exactly that.
    """
    a = dec["cards"].get(attacker_serial)
    t = dec["cards"].get(target_serial)
    if a is None or t is None:
        return None
    ac, tc = a["continual"], t["continual"]
    ps = dec["players"][attacker_pi]
    acid, tcid = a["cardId"], t["cardId"]
    atype = _type_mask(acid, ac)
    a_ability = _has_ability(acid, ac)
    active = target_area == "active"
    enemy = t["playerIndex"] != attacker_pi

    atk = a["thisTurn"]["damageChange"] + ac["damageChange"]
    if active:
        atk += a["thisTurn"]["damageChangeActive"]
    if active and enemy:
        atk += ac["damageChangeActive"] + a["turnState"]["damageChangeThisTurn"]
        atk += ps["turnState"]["playerDamageChange"]
        taken = 6 - len(((obs["current"]["players"][1 - attacker_pi]).get("prize")) or [])
        atk += ac["damageChangeEnemyTakenPrize"] * max(0, taken)
        if _is_ex(tcid):
            atk += (ac["damageChangeEx"] + a["turnState"]["damageChangeExThisTurn"]
                    + ps["turnState"]["playerDamageChangeEx"])
        if atype & _FIGHTING:
            atk += ps["turnState"]["playerDamageChangeMyFighting"]
        if ac["damageChangeAbility"] and _has_ability(tcid, tc):
            atk += ac["damageChangeAbility"]
        if ac["damageChangeEvolved"] and not _is_basic(tcid):
            atk += ac["damageChangeEvolved"]

    nete = t["nextEnemyTurnEndState"]
    tgt = (tc["takeDamageChange"] + nete["takeDamageChangeNextEnemyTurn"]
           + t["thisTurnEnemy"]["takeDamageChange"])
    zero = False
    if enemy:
        tgt += tc["takeEnemyAttackDamageChange"]
        if atype & (_FIRE | _WATER):
            tgt += tc["takeEnemyFireOrWaterPokemonAttackDamageChange"]
        if atype & (_FIRE | _WATER | _GRASS | _LIGHTNING):
            tgt += tc["takeEnemy4TypePokemonAttackDamageChange"]
        if a_ability:
            tgt += tc["takeEnemyAbilityPokemonAttackDamageChange"]
        if ps["thisTurn"]["metalDamageChange"] and (_type_mask(tcid, tc) & _METAL):
            tgt += ps["thisTurn"]["metalDamageChange"]
        zero = bool(
            tc["noDamageEnemyAttack"] or nete["noDamageAndEffectEnemyAttackNextEnemyTurn"]
            or (tc["noDamageEnemyAbilityPokemonAttack"] and a_ability)
            or (tc["noDamageEnemyExAttack"] and _is_ex(acid))
            or (tc["noDamageEnemyBasicExAttack"] and _is_basic(acid) and _is_ex(acid))
            or (tc["noDamageAndEffectEnemyTerastalAttack"]
                and getattr(_master(acid), "tera", False))
            or (t["noDamageAndEffectEnemyExAttackNextEnemyTurn"] and _is_ex(acid)))
    zero = zero or bool(
        nete["noDamageAttackNextEnemyTurn"] or nete["noDamageAndEffectAttackNextEnemyTurn"]
        or (nete["noDamageBasicAttackNextEnemyTurn"] and _is_basic(acid))
        or (nete["noDamageBasicColorAttackNextEnemyTurn"] and _is_basic(acid)
            and atype != _COLORLESS)
        or (nete["noDamageAbilityAttackNextEnemyTurn"] and a_ability))
    tm = _master(tcid)
    # Terastal takes no bench damage. That is a PRINTED property of the target, so it is left
    # out of the rendered facts (`zero_static`) -- the model can read it off the card id, and it
    # would otherwise fire on ~16% of decisions for no information.
    zero_static = bool(not active and tm is not None and getattr(tm, "tera", False))
    zero = zero or zero_static

    wk = 0
    if not nete["noWeaknessNextEnemyTurn"]:
        wmask = _ENERGY_TYPES[tc["weaknessIndex"]] if tc["weaknessIndex"] > 0 else (
            (1 << (tm.weakness - 1)) if (tm is not None and tm.weakness) else 0)
        wk = 1 if (wmask & atype) else 0
    return {"atk": atk, "tgt": tgt, "zero": zero, "zero_static": zero_static, "wk": wk,
            "nofx": bool(tc["noEffectEnemyAttack"]),
            "floor": nete["noDamageLessEqualAttackNextEnemyTurn"],
            "cap": tc["noDamageGreaterEqual"]}


def retreat_cost(dec, serial):
    """`State.h:retreatCost` exactly: printed + continual + thisTurn, zeroed by `noRetreatCost`.

    lm/costs.py derives the same number from a hand-written table of the 11 cards in the pool
    that modify it. This is the engine's own arithmetic instead, so it needs no table, cannot
    fall behind a card the pool gains, and picks up `thisTurn.retreatCostChange` -- an effect
    lm/costs.py does not model and tools/audit_costs.py structurally cannot detect, because it
    only ever RAISES the cost and the menu oracle only catches costs that are too high.
    """
    c = (dec or {}).get("cards", {}).get(serial)
    if c is None:
        return None
    m = _master(c["cardId"])
    if m is None or m.retreatCost is None:
        return None
    if c["continual"]["noRetreatCost"]:
        return 0
    return max(0, m.retreatCost + c["continual"]["retreatCostChange"]
               + c["thisTurn"]["retreatCostChange"])


# The two attack flags InsufficientEnergyCount reads that the Python card API does not expose.
# The whole 1,556-attack database has exactly one of each, and NEITHER is in any of our 60 decks
# -- listed for completeness so the port is faithful rather than approximately faithful.
# (DebugAttackEnergyFlags regenerates them.)
_NO_ENERGY_IF_SPECIAL_CONDITION = {146}     # a146 Conkeldurr
_DARKNESS1_IF_DAMAGED = {993}               # a993 Crawdaunt
_IDX_PSYCHIC, _IDX_DARKNESS = 5, 7
_IDX_ALL, _IDX_PD = 10, 11


def insufficient_energy(dec, obs, serial, attack_id, attached=None):
    """`GameUtil.h:InsufficientEnergyCount` exactly: how many MORE energies this attack needs.

    A direct port, not an approximation. The differences from the old `_shortfall` heuristic in
    lm/serialize.py are load-bearing:

      * RAINBOW is a wildcard applied to the TOTAL at the end (`allCount`), not greedily matched
        against typed symbols first -- the two disagree whenever a rainbow could pay either.
      * a typed energy that matches nothing required becomes COLORLESS payment, and colourless
        symbols are then paid by `min(colorlessCount, required.colorless)`.
      * Psychic|Darkness energy is resolved after the main pass, P first, then D, else colourless.
      * `attackCostDown` / `attackCostChangeColorless` / `thisTurn.attackCostChange` /
        `attackCostDownColorlessOwnAttack` change the cost and are invisible in the observation.

    The ATTACHED side comes from the observation, not the blob: ToJson already emits the PROVIDED
    types (`getEnergies`), so special energies that give two, or every type, are live there.
    """
    from lm import vocab
    c = (dec or {}).get("cards", {}).get(serial)
    att = vocab._ATTACKS.get(attack_id)
    if c is None or att is None:
        return None
    master = _master(c["cardId"])
    if master is None:
        return None
    cont = c["continual"]
    own = attack_id in (master.attacks or [])
    pi = c["playerIndex"]
    try:
        pl = obs["current"]["players"][pi]
    except (KeyError, IndexError, TypeError):
        return None
    if attached is None:
        for _p, _z, _i, ser, m in in_play_serials(obs):
            if ser == serial:
                attached = m.get("energies") or []
                break
    if attached is None:
        return None

    req = [0] * 12
    req_colorless = 0
    req_sum = 0
    fix = False
    special = any(pl.get(k) for k in ("poisoned", "burned", "asleep", "paralyzed", "confused"))
    if attack_id in _NO_ENERGY_IF_SPECIAL_CONDITION and special:
        fix = True
    elif (cont["attackEnergyColoressOne"] or cont["attackEnergyPsychicOne"]) and own:
        fix = True
        if cont["attackEnergyColoressOne"]:
            req_colorless += 1
        else:
            req[_IDX_PSYCHIC] += 1
        req_sum += 1
    elif attack_id in _DARKNESS1_IF_DAMAGED and c["damage"] > 0:
        fix = True
        req[_IDX_DARKNESS] += 1
        req_sum += 1
    else:
        for e in (att.energies or []):
            if e == 0:
                req_colorless += 1
            else:
                req[e] += 1
            req_sum += 1

    all_count = pd_count = colorless_count = 0
    for e in attached:
        if e == _IDX_ALL:
            all_count += 1
        elif e == 0:
            colorless_count += 1
        elif e == _IDX_PD:
            pd_count += 1
        elif req[e] > 0:
            req[e] -= 1
            req_sum -= 1
        else:
            colorless_count += 1

    if not fix:
        all_count += cont["attackCostDown"]
        change = cont["attackCostChangeColorless"] + c["thisTurn"]["attackCostChange"]
        if change < 0:
            colorless_count -= change
        else:
            req_colorless += change
            req_sum += change
        if cont["attackCostDownColorlessOwnAttack"] > 0 and own:
            colorless_count += cont["attackCostDownColorlessOwnAttack"]

    for _ in range(pd_count):
        if req[_IDX_PSYCHIC] > 0:
            req[_IDX_PSYCHIC] -= 1
            req_sum -= 1
        elif req[_IDX_DARKNESS] > 0:
            req[_IDX_DARKNESS] -= 1
            req_sum -= 1
        else:
            colorless_count += 1

    return max(0, req_sum - all_count - min(colorless_count, req_colorless))


def need_energy(dec, obs, serial, attached=None):
    """Smallest number of extra energies that makes ANY DAMAGING attack payable -- the same
    quantity `need:N` has always rendered, computed with the engine's arithmetic.

    ``attached`` overrides the energy list read from the observation -- used to answer "and
    AFTER this attach?" for the menu annotation."""
    c = (dec or {}).get("cards", {}).get(serial)
    if c is None:
        return None
    from lm import vocab
    master = _master(c["cardId"])
    if master is None or not master.attacks:
        return None
    vals = []
    for aid in master.attacks:
        a = vocab._ATTACKS.get(aid)
        if a is None or not a.damage:
            continue
        v = insufficient_energy(dec, obs, serial, aid, attached=attached)
        if v is not None:
            vals.append(v)
    return min(vals) if vals else None


def board_extra(obs, dec=None):
    """`{serial: " dmg:+30"}`-style suffixes for lm/serialize.py, rendered only when non-default.

    Facts are relative to the ACTING player's Active attacker, because that is the pair every
    attack option on the menu concerns. Empty dict when nothing is modified -- which the audit
    says is ~82% of decisions, so the average token cost is a fraction of the worst case.
    """
    if dec is None:
        dec = read(obs)
    if dec is None:
        return {}
    cur = obs.get("current") or {}
    yi = cur.get("yourIndex", 0)
    slots = in_play_serials(obs)
    mine = [s for pi, z, i, s, m in slots if pi == yi and z == "A"]
    if not mine:
        return {}
    me = mine[0]
    thr = active_threat(obs, dec, yi)
    my_hp = next((m.get("hp") or 0 for pi, z, i, s, m in slots if s == me), 0)
    out = {}
    for pi, z, i, serial, m in slots:
        bits = []
        if serial == me:
            f = attack_facts(obs, dec, yi, me, serial, "active")
        elif pi != yi:
            f = attack_facts(obs, dec, yi, me, serial, "active" if z == "A" else "bench")
        else:
            f = None
        if f is None:
            continue
        if serial == me:
            if f["atk"]:
                bits.append("dmg:%+d" % f["atk"])
            if thr is not None:
                val, exact = thr
                # `!` = would KO our Active as the board stands. Only on an exact number.
                bits.append("thr:%d%s" % (val, "!" if exact and val >= my_hp else "")
                            if exact else "thr~%d" % val)
        else:
            if f["zero"] and not f["zero_static"]:
                bits.append("tk:x0")
            elif f["tgt"]:
                bits.append("tk:%+d" % f["tgt"])
            if f["nofx"]:
                bits.append("fx:x")
            if f["floor"]:
                bits.append("tk<=%d:0" % f["floor"])
        if bits:
            out[serial] = " " + " ".join(bits)
    return out


# ------------------------------------------------------------------------------------------
# Menu-facing derived facts: post-attach need, and the threat on our Active.
# ------------------------------------------------------------------------------------------

# What each Special Energy PROVIDES, as energy-type indexes (State.h:getEnergyInfo). 10 = every
# type ("rainbow"), 11 = Psychic|Darkness. Conditional providers (Neo Upper / Prism / Ignition)
# are handled in provided_energy; anything not listed there fails closed to "no annotation".
# tools/verify_menu_notes.py checks every prediction against the energies the engine actually
# attached, which is what makes a hand-written table safe to ship.
_SPECIAL_PROVIDES = {
    9: [0],          # Boomerang
    11: [0],         # Mist
    13: [0],         # Enriching
    14: [0],         # Spiky
    18: [1],         # Grow Grass
    19: [5],         # Telepath Psychic
    20: [6],         # Rock Fighting
    12: [10],        # Legacy: every type, 1 at a time
    15: [11, 11],    # Team Rocket's: 2 in any combination of {P} and {D}
}
_KIND_BASIC_ENERGY, _KIND_SPECIAL_ENERGY = 5, 6


_GRASS_ENERGY_CID = 1


def provided_energy(energy_cid, target_cid, target_continual=None):
    """Energy-type indexes this card provides once attached to `target_cid`, or None.

    Mirrors State.h:getEnergyInfo exactly, including the two target-dependent rules the first
    version missed and tools/verify_menu_notes.py caught: a Basic {G} Energy provides G x2 on a
    Pokemon carrying the `doubleGrassEnergy` continual flag (an ability, so it needs the decoded
    hidden state), and Ignition's "Evolution" means Stage 1/2 ONLY -- the engine tests
    `evolutionType`, not the Mega flag.
    """
    from lm import vocab
    c = vocab._CARDS.get(energy_cid)
    if c is None:
        return None
    if c.cardType == _KIND_BASIC_ENERGY:
        if (energy_cid == _GRASS_ENERGY_CID and target_continual is not None
                and target_continual.get("doubleGrassEnergy")):
            return [1, 1]
        return [c.energyType]
    if c.cardType != _KIND_SPECIAL_ENERGY:
        return None
    tm = vocab._CARDS.get(target_cid)
    if energy_cid == 10:      # Neo Upper: Stage 2 -> every type x2, else C
        return [10, 10] if (tm is not None and tm.stage2) else [0]
    if energy_cid == 16:      # Prism: Basic -> every type, else C
        return [10] if (tm is not None and tm.basic) else [0]
    if energy_cid == 17:      # Ignition: Stage 1/2 -> CCC, else C
        return [0, 0, 0] if (tm is not None and (tm.stage1 or tm.stage2)) else [0]
    return _SPECIAL_PROVIDES.get(energy_cid)


def post_attach_need(obs, dec, target_serial, energy_cid):
    """`need` of the target AFTER attaching `energy_cid`, or None when not derivable."""
    target = None
    for _p, _z, _i, ser, m in in_play_serials(obs):
        if ser == target_serial:
            target = m
            break
    if target is None:
        return None
    tc = (dec or {}).get("cards", {}).get(target_serial)
    prov = provided_energy(energy_cid, target.get("id"),
                           tc["continual"] if tc else None)
    if prov is None:
        return None
    return need_energy(dec, obs, target_serial,
                       attached=list(target.get("energies") or []) + prov)


def active_threat(obs, dec, yi):
    """(damage, exact) -- the most the opponent's Active could land on ours NEXT TURN with the
    energy it has now, through the full damage pipeline. None when there is nothing to compute.

    A LOWER BOUND by construction: it cannot see the attach/switch/pump plays in the opponent's
    hand. `exact` is False when a coin expectation or an unevaluable payable attack is involved
    -- rendered `thr~`, never a certain number.
    """
    from lm import damage, vocab
    cur = obs.get("current") or {}
    players = cur.get("players") or []
    if len(players) != 2:
        return None
    op = 1 - yi
    try:
        opp_act = (players[op].get("active") or [None])[0]
        my_act = (players[yi].get("active") or [None])[0]
    except (IndexError, TypeError):
        return None
    if not opp_act or not my_act:
        return None
    master = vocab._CARDS.get(opp_act.get("id"))
    if master is None or not master.attacks:
        return None
    best, exact, any_payable = None, True, False
    for aid in master.attacks:
        ins = insufficient_energy(dec, obs, opp_act["serial"], aid)
        if ins is None or ins > 0:
            continue
        any_payable = True
        val, kind = damage.final_damage(obs, dec, opp_act["serial"], my_act["serial"], aid, op)
        if val is None:
            exact = False        # a payable attack we cannot price -> the bound is soft
            continue
        if kind == "expected":
            if best is None or val > best:
                best, exact = val, False
        elif best is None or val > best:
            best = val
    if not any_payable or best is None:
        return None
    return (best, exact)
