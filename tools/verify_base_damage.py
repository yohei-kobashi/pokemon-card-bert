#!/usr/bin/env python3
"""Did lm/damage.py predict the base damage the engine actually used?

The dynamic part of an attack's damage exists only while the attack resolves, so it cannot be
queried from outside. build_engine_mirror.py patches GameProc.h to stash the last
`attack.damage + state.attackDamageChange`; this predicts the value at the moment the attack was
still a MENU ENTRY, lets the game play on, and compares.

Reports three things, all of which matter:
  * accuracy on the predictions it makes (must be 100% for "exact")
  * COVERAGE -- the share of resolved attacks it declined to predict, which is the real cost of
    a partial evaluator
  * the attacks it gets wrong or skips, so the next family to implement is obvious

    python3 tools/verify_base_damage.py --games 6
"""

import argparse
import collections
import ctypes
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "cg-lib")):
    if p not in sys.path:
        sys.path.insert(0, p)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--decks", default="")
    ap.add_argument("--games", type=int, default=6)
    ap.add_argument("--seed-base", type=int, default=770000)
    ap.add_argument("--opp", default="live")
    ap.add_argument("--shard", default="")
    ap.add_argument("--so", default=os.path.join(ROOT, "data", "kaggle_engine_ext",
                                                 "libcg_hidden.so"))
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    import library
    from lm import damage, hidden, vocab
    from lm.actions import encode_option
    from lm.agent import make_lm_agent
    from tools.mirror_env import MirrorEngine

    eng = MirrorEngine(a.so)
    eng.lib.DebugLastBaseDamage.argtypes = [ctypes.POINTER(ctypes.c_int)]
    buf = (ctypes.c_int * 4)()

    def last_base():
        eng.lib.DebugLastBaseDamage(buf)
        return list(buf)

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
    wrong = collections.Counter()
    skipped = collections.Counter()
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
            pending = {}
            prev = last_base()
            try:
                for _ in range(4000):
                    cur = obs.get("current") or {}
                    if cur.get("result", -1) != -1 or not obs.get("select"):
                        break
                    now = last_base()
                    if now != prev and now[0]:
                        aid, serial, base, printed = now
                        key = (aid, serial)
                        if key in pending:
                            pred, kind = pending.pop(key)
                            st["resolved"] += 1
                            if pred is None:
                                st["skipped"] += 1
                                skipped[aid] += 1
                            elif kind == "expected":
                                st["expected"] += 1
                            elif pred == base:
                                st["exact_ok"] += 1
                            else:
                                st["exact_bad"] += 1
                                wrong[aid] += 1
                                if len(examples) < 12:
                                    examples.append((deck, aid, printed, base, pred))
                        pending.clear()
                    prev = now
                    yi = cur.get("yourIndex", 0)
                    dec = hidden.read(obs)
                    if dec is not None:
                        for o in ((obs.get("select") or {}).get("option") or []):
                            t = encode_option(o, obs)
                            if not t.startswith("attack:"):
                                continue
                            try:
                                aid = int(t.split(":")[1])
                            except ValueError:
                                continue
                            act = (cur["players"][yi].get("active") or [None])[0]
                            if not act:
                                continue
                            pred = damage.base_damage(obs, dec, act["serial"], aid, yi)
                            pending[(aid, act["serial"])] = pred
                            # A: the FINAL number the menu renders, vs the engine's own
                            # CalcDamage run at the same base. Also cross-checks the baked
                            # noTarget* flags against the ones the engine reports.
                            opp = (cur["players"][1 - yi].get("active") or [None])[0]
                            if pred[0] is not None and pred[1] == "exact" and opp:
                                fin, fkind = damage.final_damage(
                                    obs, dec, act["serial"], opp["serial"], aid, yi)
                                truth = eng.calc_damage(act["serial"], opp["serial"],
                                                        pred[0], aid)
                                if truth is not None:
                                    tflags = (truth["noTargetEffect"],
                                              truth["noTargetWeakness"],
                                              truth["noTargetResistance"])
                                    if tuple(damage.flags(aid)) != tflags:
                                        st["flags_bad"] += 1
                                    if fin is None:
                                        st["final_skipped"] += 1
                                    elif fin == truth["damage"]:
                                        st["final_ok"] += 1
                                    else:
                                        st["final_bad"] += 1
                                        wrong[aid] += 1
                                        if len(examples) < 12:
                                            examples.append((deck, aid, pred[0],
                                                             truth["damage"], fin))
                    obs = eng.select((agent if yi == 0 else oagent)(obs))
            except Exception:
                pass
            finally:
                eng.finish()

    n = st["resolved"] or 1
    pred = st["exact_ok"] + st["exact_bad"]
    print("\n%d resolved attacks matched to a menu prediction" % st["resolved"])
    print("  exact predicted   %6d (%5.1f%%)   correct %d, WRONG %d"
          % (pred, 100 * pred / n, st["exact_ok"], st["exact_bad"]))
    print("  expected (coin)   %6d (%5.1f%%)   not checkable per-sample" % (st["expected"],
                                                                           100 * st["expected"] / n))
    print("  declined          %6d (%5.1f%%)" % (st["skipped"], 100 * st["skipped"] / n))
    fn = st["final_ok"] + st["final_bad"] or 1
    print("FINAL damage vs CalcDamage: %d checked, %d ok, WRONG %d, skipped %d | flag "
          "mismatches %d" % (st["final_ok"] + st["final_bad"], st["final_ok"],
                             st["final_bad"], st["final_skipped"], st["flags_bad"]))
    if wrong:
        print("\nWRONG (implement or drop these):")
        for aid, c in wrong.most_common(15):
            at = vocab._ATTACKS.get(aid)
            print("  a%-6d %-26s %d" % (aid, (at.name if at else "?")[:26], c))
        for deck, aid, printed, base, p in examples:
            print("    %-18s a%-6d printed %4d engine %4d predicted %4d"
                  % (deck, aid, printed, base, p))
    if skipped:
        print("\ndeclined, most frequent first (the next family to implement):")
        tbl = damage.table()
        for aid, c in skipped.most_common(12):
            at = vocab._ATTACKS.get(aid)
            tags = ",".join((tbl.get(aid) or {}).get("tags") or [])
            print("  a%-6d %-26s %6d   %s" % (aid, (at.name if at else "?")[:26], c, tags))
    if a.out:
        json.dump({"st": dict(st), "wrong": {str(k): v for k, v in wrong.items()},
                   "skipped": {str(k): v for k, v in skipped.items()}}, open(a.out, "w"))
    return 1 if st["exact_bad"] or st["final_bad"] or st["flags_bad"] else 0


if __name__ == "__main__":
    sys.exit(main())
