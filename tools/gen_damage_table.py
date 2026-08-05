#!/usr/bin/env python3
"""Generate lm/damage_table.json -- the damage formula of every dynamic attack in our decks.

`GameProc.h:AttackDamage` uses `attack.damage + state.attackDamageChange`, and the second term is
written by the attack's own effects during resolution. The printed `damage` is therefore only a
floor for those attacks, and with `glossary="none"` the prompt does not even carry the printed
number -- the model has only the `a1234` token, which cannot encode a value that moves.

This dumps the effects, field by field, from the instrumented build (DebugDamageFormula), so the
evaluator in lm/damage.py is written against what the effect DOES rather than against the card's
English text. Regenerate after an engine refetch.

    python3 tools/gen_damage_table.py                 # attacks in decks/ only
    python3 tools/gen_damage_table.py --all           # every attack in the database
"""

import argparse
import ctypes
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "cg-lib")):
    if p not in sys.path:
        sys.path.insert(0, p)

SO = os.path.join(ROOT, "data", "kaggle_engine_ext", "libcg_hidden.so")
OUT = os.path.join(ROOT, "lm", "damage_table.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--so", default=SO)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    import library
    from lm import vocab

    lib = ctypes.cdll.LoadLibrary(a.so)
    lib.GameInitialize()
    for fn in ("DebugDamageDynamics", "DebugDamageFormula"):
        getattr(lib, fn).restype = ctypes.c_char_p
    lib.DebugDamageFormula.argtypes = [ctypes.c_int]

    dyn = {int(k): v for k, v in json.loads(lib.DebugDamageDynamics().decode()).items()}
    keep = set()
    if a.all:
        keep = set(dyn)
    else:
        pool = set()
        for d in sorted(library.list_decks()):
            for line in open(library.deck_path(d)):
                if line.strip():
                    pool.add(int(line))
        for cid in pool:
            c = vocab._CARDS.get(cid)
            for aid in ((c.attacks if c else None) or []):
                if aid in dyn:
                    keep.add(aid)

    out = {}
    for aid in sorted(keep):
        at = vocab._ATTACKS.get(aid)
        out[str(aid)] = {
            "printed": at.damage if at else 0,
            "tags": dyn[aid],
            "effects": json.loads(lib.DebugDamageFormula(aid).decode()),
        }
    json.dump(out, open(a.out, "w"), indent=1, sort_keys=True)
    print("wrote %s: %d attacks" % (a.out, len(out)))


if __name__ == "__main__":
    main()
