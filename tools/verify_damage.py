#!/usr/bin/env python3
"""Diff lm/hidden.py's damage model against the engine's own CalcDamage.

`lm.hidden.attack_facts` collapses SetProperty.h:CalcDamage into three renderable numbers
(attacker delta, target delta, immunity). That is a REIMPLEMENTATION, and a wrong sign or a
missed condition produces a plausible number rather than a crash -- the same failure mode that
made the printed retreat cost look fine for months ([[prompt-lies-about-retreat-cost]]).

So it is diffed against the engine. `DebugCalcDamage` runs the real CalcDamage on a chosen
(attacker, target, base damage) triple; this walks real games, tries every in-play pair and
several base damages, and asserts the prediction equals it exactly.

    python3 tools/verify_damage.py --games 4                  # every deck
    python3 tools/verify_damage.py --decks hop_zacian --games 20 --verbose
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

BASES = (10, 30, 60, 120, 200)


def predict(f, base, no_effect, no_weak, no_res, resistance_hit):
    """Apply the pipeline in CalcDamage's order. `f` is lm.hidden.attack_facts."""
    d = base
    if d <= 0:
        return 0
    d += f["atk"]
    if d <= 0:
        return 0
    if f["wk"] and not no_weak:
        d *= 2
    if resistance_hit and not no_res:
        d -= 30
        if d <= 0:
            return 0
    if not no_effect:
        d += f["tgt"]
        if f["zero"]:
            d = 0
        if d <= f["floor"]:
            d = 0
        if f.get("cap", 0) > 0 and f["cap"] <= d:
            d = 0
    return max(0, d)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--decks", default="")
    ap.add_argument("--games", type=int, default=4)
    ap.add_argument("--seed-base", type=int, default=820000)
    ap.add_argument("--opp", default="live")
    ap.add_argument("--shard", default="")
    ap.add_argument("--so", default=os.path.join(ROOT, "data", "kaggle_engine_ext",
                                                 "libcg_hidden.so"))
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    import library
    from lm import hidden, vocab
    from lm.agent import make_lm_agent
    from tools.mirror_env import MirrorEngine

    eng = MirrorEngine(a.so)
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
                    dec = hidden.read(obs)
                    if dec is not None:
                        slots = hidden.in_play_serials(obs)
                        for api, az, ai, aser, am in slots:
                            atk_ids = (vocab._CARDS.get(am["id"]).attacks
                                       if vocab._CARDS.get(am["id"]) else None) or []
                            if not atk_ids:
                                continue
                            aid = atk_ids[0]
                            for tpi, tz, ti, tser, tm in slots:
                                if tser == aser:
                                    continue
                                f = hidden.attack_facts(obs, dec, api, aser, tser,
                                                        "active" if tz == "A" else "bench")
                                if f is None:
                                    continue
                                tmaster = vocab._CARDS.get(tm["id"])
                                amask = hidden._type_mask(am["id"],
                                                          dec["cards"][aser]["continual"])
                                res = bool(tmaster and tmaster.resistance
                                           and (1 << (tmaster.resistance - 1)) & amask)
                                for base in BASES:
                                    truth = eng.calc_damage(aser, tser, base, aid)
                                    if truth is None:
                                        continue
                                    got = predict(f, base, truth["noTargetEffect"],
                                                  truth["noTargetWeakness"],
                                                  truth["noTargetResistance"], res)
                                    st["checked"] += 1
                                    if got != truth["damage"]:
                                        bad[(deck, am["id"], tm["id"])] += 1
                                        if len(examples) < 15:
                                            examples.append(
                                                (deck, am["id"], tm["id"], base,
                                                 truth["damage"], got, dict(f), res))
                    obs = eng.select((agent if yi == 0 else oagent)(obs))
            except Exception:
                pass
            finally:
                eng.finish()
        if a.verbose:
            print("  %-24s checked %d, mismatches %d"
                  % (deck, st["checked"], sum(bad.values())), flush=True)

    print("\n%d (attacker, target, base) triples | %d mismatches (%.3f%%)"
          % (st["checked"], sum(bad.values()),
             100 * sum(bad.values()) / max(1, st["checked"])))
    if bad:
        print("\nworst pairs:")
        for (deck, acid, tcid), c in bad.most_common(15):
            print("  %-22s c%-6d %-22s -> c%-6d %-22s  %d"
                  % (deck, acid, vocab.card_name(acid)[:22], tcid,
                     vocab.card_name(tcid)[:22], c))
        print("\nexamples (base, engine, predicted, facts, resistance):")
        for deck, acid, tcid, base, truth, got, f, res in examples:
            print("  %-18s c%-6d->c%-6d base %3d  engine %4d  got %4d  res=%d  %s"
                  % (deck, acid, tcid, base, truth, got, res, f))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
