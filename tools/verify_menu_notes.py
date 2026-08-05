#!/usr/bin/env python3
"""Check the attach annotation (` n:K`) against what the engine actually attaches.

Two claims go into `n:K` and both are hand-written, so both are checked in real games:

  provided   lm/hidden.provided_energy says what energy types the card gives once attached
             (incl. the conditional Special Energies: Neo Upper / Prism / Ignition / TR).
             Oracle: whenever ANY attach resolves, the target's `energies` list in the next
             observation must equal the old list plus exactly the predicted types.
  need       the post-attach `need` predicted from that list must equal the need recomputed
             from the REAL post-attach observation.

    python3 tools/verify_menu_notes.py --games 6
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


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--decks", default="")
    ap.add_argument("--games", type=int, default=6)
    ap.add_argument("--seed-base", type=int, default=880000)
    ap.add_argument("--opp", default="live")
    ap.add_argument("--shard", default="")
    ap.add_argument("--so", default=os.path.join(ROOT, "data", "kaggle_engine_ext",
                                                 "libcg_hidden.so"))
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    import library
    from lm import hidden, vocab
    from lm.actions import encode_option
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

    def energies_of(obs, serial):
        for _p, _z, _i, ser, m in hidden.in_play_serials(obs):
            if ser == serial:
                return list(m.get("energies") or [])
        return None

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
                    choice = (agent if yi == 0 else oagent)(obs)
                    check = None
                    opts = (obs.get("select") or {}).get("option") or []
                    if len(choice) == 1 and 0 <= choice[0] < len(opts):
                        o = opts[choice[0]]
                        t = encode_option(o, obs)
                        if t.startswith("attach:c"):
                            key = {4: "active", 5: "bench"}.get(o.get("inPlayArea"))
                            idx = o.get("inPlayIndex") or 0
                            try:
                                tgt = (cur["players"][yi].get(key) or [])[idx]
                            except (KeyError, IndexError, TypeError):
                                tgt = None
                            if tgt and tgt.get("serial") is not None:
                                ecid = int(t.split(":c")[1].split("@")[0])
                                dec = hidden.read(obs)
                                tc = (dec or {}).get("cards", {}).get(tgt["serial"])
                                prov = hidden.provided_energy(
                                    ecid, tgt.get("id"), tc["continual"] if tc else None)
                                pred_need = (hidden.post_attach_need(obs, dec, tgt["serial"],
                                                                    ecid)
                                             if dec is not None else None)
                                check = (tgt["serial"], ecid, tgt.get("id"),
                                         list(tgt.get("energies") or []), prov, pred_need)
                    obs = eng.select(choice)
                    if check is None:
                        continue
                    serial, ecid, tcid, before, prov, pred_need = check
                    after = energies_of(obs, serial)
                    if after is None:       # target left play mid-resolution; cannot judge
                        continue
                    st["attaches"] += 1
                    if prov is None:
                        st["provided_skipped"] += 1
                    else:
                        st["provided_checked"] += 1
                        if sorted(before + prov) != sorted(after):
                            st["provided_bad"] += 1
                            bad[("provided", ecid)] += 1
                            if len(examples) < 10:
                                examples.append((deck, ecid, tcid, before, prov, after))
                    if pred_need is not None:
                        dec2 = hidden.read(obs)
                        actual = (hidden.need_energy(dec2, obs, serial)
                                  if dec2 is not None else None)
                        if actual is not None:
                            st["need_checked"] += 1
                            if pred_need != actual:
                                st["need_bad"] += 1
                                bad[("need", ecid)] += 1
                                if len(examples) < 10:
                                    examples.append((deck, ecid, tcid, pred_need, actual,
                                                     after))
            except Exception:
                pass
            finally:
                eng.finish()

    print("\n%d attaches observed" % st["attaches"])
    print("  provided-energy checked %d, WRONG %d, skipped (fail-closed) %d"
          % (st["provided_checked"], st["provided_bad"], st["provided_skipped"]))
    print("  post-attach need checked %d, WRONG %d" % (st["need_checked"], st["need_bad"]))
    if bad:
        from lm import vocab
        print("\nmismatches:")
        for (kind, ecid), c in bad.most_common(10):
            print("  %-9s c%-6d %-26s %d" % (kind, ecid, vocab.card_name(ecid)[:26], c))
        for e in examples:
            print("   ", e)
    if a.out:
        json.dump({"st": dict(st),
                   "bad": {"|".join(map(str, k)): v for k, v in bad.items()}}, open(a.out, "w"))
    return 1 if st["provided_bad"] or st["need_bad"] else 0


if __name__ == "__main__":
    sys.exit(main())
