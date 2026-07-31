"""P0.5 line-config generator (pipeline v2.1): derive a machine-readable L2
line config for ANY deck from the card DB alone — no per-deck code, no LLM.
The generic engine_v2.ConfigL2 consumes it; P1-P4 then measure/accept it as a
bundle. This automates the mechanical part of P0 (focus ranking, stadiums,
bench width, discard-fuel edges); judgment-heavy lines (rotations, multi-body
combos) still come from the full P0' analysis.

Usage:  PYTHONPATH=cg-lib python tools/p05_deckconfig.py <deck> [--write]
  --write  merge the config into agents/tuning.json (l2="config", line={...})
"""
import os, sys, json, argparse, re
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "cg-lib"), os.path.join(ROOT, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)
import library
from collections import Counter
from cg.api import CardType, EnergyType
from agents._engine import _CARDS, _ATTACKS
from agents.engine_v2 import _atk_value

_LETTER = {"G": EnergyType.GRASS, "R": EnergyType.FIRE, "W": EnergyType.WATER,
           "L": EnergyType.LIGHTNING, "P": EnergyType.PSYCHIC,
           "F": EnergyType.FIGHTING, "D": EnergyType.DARKNESS,
           "M": EnergyType.METAL}
_RE_DISCARD_ATTACH = re.compile(
    r"attach (?:a|up to \d+) basic {(\w)} energy card(?:s)? from your discard",
    re.I)
_RE_BENCH_WIDE = re.compile(r"up to (\d+) pok\S*mon on their bench", re.I)
_RE_EXACTKO = re.compile(r"exactly (\d+) damage counters.{0,40}knocked out", re.I | re.S)
_RE_MOVER = re.compile(
    r"move up to (\d+) damage counters from 1 of your pok\S*mon to 1 of your opponent",
    re.I)
_RE_OPPDMG_BONUS = re.compile(
    r"opponent.{1,3}s active pok\S*mon already has any damage counters on it, "
    r"this attack does (\d+) more damage", re.I)


_RE_EACH_BENCH = re.compile(r"(\d+)\s*(?:more\s+)?damage\s+for\s+each\s+benched", re.I)


def deck_value(a, bench_w):
    """Attack value in DECK CONTEXT: benched-count scalers are evaluated at the
    deck's own bench width (a bench-widening stadium changes the whole line —
    the lillies Rondo case, generalized)."""
    v = _atk_value(a)
    m = _RE_EACH_BENCH.search(a.text or "")
    if m:
        mine = bench_w or 5
        n = mine + 5 if "both" in (a.text or "").lower() else mine
        v = (a.damage or 0) + int(m.group(1)) * n
    return v


def deck_energy_types(counts):
    out = set()
    for cid in counts:
        c = _CARDS[cid]
        if c.cardType == CardType.BASIC_ENERGY:
            out.add(c.energyType)
        elif c.cardType == CardType.SPECIAL_ENERGY:
            t = (c.skills[0].text if c.skills else "") or ""
            for L, e in _LETTER.items():
                if "{%s}" % L in t:
                    out.add(e)
    return out

def payable(atk, etypes):
    """Typed costs must be coverable by the deck's energy types (C = any)."""
    for e in (atk.energies or []):
        if e != EnergyType.COLORLESS and e not in etypes:
            return False
    return True


def generate(deck_name):
    counts = Counter(library.read_deck(deck_name))
    etypes = deck_energy_types(counts)
    # ---- stadiums / bench width (needed BEFORE ranking: deck context) --------
    stadiums, bench_target = [], None
    for cid in counts:
        c = _CARDS[cid]
        if c.cardType == 4:
            stadiums.append(cid)
            t = (c.skills[0].text if c.skills else "") or ""
            m = _RE_BENCH_WIDE.search(t)
            if m:
                bench_target = int(m.group(1))
    # ---- focus ranking: value-per-energy of each body's best PAYABLE attack --
    rows = []
    for cid, copies in counts.items():
        c = _CARDS[cid]
        if c.cardType != CardType.POKEMON:
            continue
        best = None
        for aid in (c.attacks or []):
            a = _ATTACKS.get(aid)
            if a is None or not payable(a, etypes):
                continue
            v = deck_value(a, bench_target)
            cost = max(1, len(a.energies or []))
            if v and (best is None or v / cost > best[0]):
                best = (v / cost, v, cost, a.attackId)
        if best is None:
            continue
        vpc, v, cost, aid = best
        # weight: efficiency x raw value, small bonus for copies (consistency)
        rows.append({"cid": cid, "name": c.name, "score": vpc * (v ** 0.5) * (1 + 0.1 * copies),
                     "value": v, "need": cost, "attack": aid, "copies": copies})
    rows.sort(key=lambda r: -r["score"])
    focus = rows[:3]
    # ---- discard-fuel edges (attach-from-discard trainers) -------------------
    fuel = set()
    for cid in counts:
        c = _CARDS[cid]
        if c.cardType in (CardType.POKEMON, CardType.BASIC_ENERGY,
                          CardType.SPECIAL_ENERGY):
            continue
        t = (c.skills[0].text if c.skills else "") or ""
        m = _RE_DISCARD_ATTACH.search(t)
        if m:
            e = _LETTER.get(m.group(1).upper())
            for ec in counts:
                cc = _CARDS[ec]
                if cc.cardType == CardType.BASIC_ENERGY and cc.energyType == e:
                    fuel.add(ec)
    # ---- exact-counters combo (v2.3): conditional-KO finisher + counter movers
    combo = {}
    support_energy = {}
    from agents._engine import _ATTACKS as _AT
    for cid in counts:
        c = _CARDS[cid]
        if c.cardType != CardType.POKEMON:
            continue
        for aid in (c.attacks or []):
            a = _AT.get(aid)
            m = _RE_EXACTKO.search((a.text or "")) if a else None
            if m and payable(a, etypes):
                combo = {"type": "exact_counters", "n": int(m.group(1)),
                         "finisher": cid, "finisher_attack": a.attackId}
    support_etype = {}
    for cid in counts:
        c = _CARDS[cid]
        if c.cardType != CardType.POKEMON:
            continue
        for sk in (c.skills or []):
            if _RE_MOVER.search(sk.text or ""):
                # movers pay for themselves even without a combo finisher:
                # +30 chip/turn to the opponent AND 30 effective heal/turn
                support_energy[str(cid)] = 1
                m2 = re.search(r"has any {(\w)} energy", sk.text or "", re.I)
                if m2:
                    e = _LETTER.get(m2.group(1).upper())
                    if e is not None:
                        support_etype[str(cid)] = int(e)
    cfg = {
        "focus": [r["cid"] for r in focus],
        "need": {str(r["cid"]): r["need"] for r in focus},
        "big": sorted({r["attack"] for r in focus}),
        "stadiums": stadiums,
        "discard_fuel": sorted(fuel),
    }
    # v2.4: does any payable deck attack gain a bonus vs a DAMAGED opp active?
    predamage = False
    for cid in counts:
        c = _CARDS[cid]
        if c.cardType != CardType.POKEMON:
            continue
        for aid in (c.attacks or []):
            a = _AT.get(aid)
            if a is not None and payable(a, etypes) and _RE_OPPDMG_BONUS.search(a.text or ""):
                predamage = True
    if predamage:
        cfg["predamage"] = True
    if combo:
        cfg["combo"] = combo
    if support_energy:
        cfg["support_energy"] = support_energy
    if support_etype:
        cfg["support_etype"] = support_etype
    if bench_target:
        cfg["bench_target"] = bench_target
    report = {"deck": deck_name,
              "energy_types": sorted(int(e) for e in etypes),
              "focus_ranking": rows[:6], "config": cfg}
    return cfg, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("deck")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--support-only", action="store_true",
                    help="emit only support/mover/predamage keys; keep the deck's "
                         "archetype pilot (l2 unchanged) — for decks where the "
                         "focus doctrine is measured harmful")
    args = ap.parse_args()
    cfg, report = generate(args.deck)
    print(json.dumps(report, ensure_ascii=False, indent=1))
    if args.write:
        tp = os.path.join(ROOT, "agents", "tuning.json")
        tun = json.load(open(tp))
        if args.support_only:
            slim = {k: cfg[k] for k in ("support_energy", "support_etype", "predamage")
                    if k in cfg}
            tun.setdefault(args.deck, {})["line"] = slim
            print(f"-- written: {args.deck} support-only line={slim} (l2 unchanged)")
        else:
            tun.setdefault(args.deck, {})["l2"] = "config"
            tun[args.deck]["line"] = cfg
            print(f"-- written to tuning.json: {args.deck} l2=config")
        json.dump(tun, open(tp, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
