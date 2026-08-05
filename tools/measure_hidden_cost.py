#!/usr/bin/env python3
"""Token cost of putting the engine's hidden effect state in the prompt.

`hidden_facts=True` adds, per Pokemon and only when non-default, `dmg:+N` on our attacker and
`tk:+N` / `tk:x0` / `fx:x` on what it would hit -- the collapsed form of the hidden damage model
(lm/hidden.py). Deploy speed is the binding constraint on this whole track
([[rerank-deploy-quantization-and-speed]]), so the cost is measured with the SAME tokenizer the
model uses, not estimated from characters.

    python3 tools/measure_hidden_cost.py --tokenizer /path/to/tokenizer.json --games 8
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
    ap.add_argument("--tokenizer", required=True, help="tokenizer.json (the trained one)")
    ap.add_argument("--decks", default="")
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--seed-base", type=int, default=610000)
    ap.add_argument("--opp", default="live")
    ap.add_argument("--shard", default="")
    ap.add_argument("--so", default=os.path.join(ROOT, "data", "kaggle_engine_ext",
                                                 "libcg_hidden.so"))
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    import library
    from tokenizers import Tokenizer
    from lm.agent import make_lm_agent
    from lm.serialize import serialize_stateless
    from tools import rl_config
    from tools.mirror_env import MirrorEngine

    tok = Tokenizer.from_file(a.tokenizer)
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
    fmt = dict(rl_config.PROMPT_FMT)

    def load(n):
        return [int(x) for x in open(library.deck_path(n)) if x.strip()]

    base_tok, new_tok, delta = [], [], []
    st = collections.Counter()
    kinds = collections.Counter()
    per_deck = {}
    for di, deck in enumerate(decks):
        ids = load(deck)
        agent = make_lm_agent(ids, tuning.get(deck, {}), model=None)
        d_delta = []
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
                    if yi == 0 and len((obs["select"].get("option") or [])) >= 2:
                        p0 = serialize_stateless(obs, deck_ids=ids, deck_name=deck, **fmt)
                        p1 = serialize_stateless(obs, deck_ids=ids, deck_name=deck,
                                                 hidden_facts=True, **fmt)
                        n0 = len(tok.encode(p0).ids)
                        st["decisions"] += 1
                        base_tok.append(n0)
                        if p1 == p0:
                            new_tok.append(n0)
                            delta.append(0)
                            d_delta.append(0)
                        else:
                            n1 = len(tok.encode(p1).ids)
                            new_tok.append(n1)
                            delta.append(n1 - n0)
                            d_delta.append(n1 - n0)
                            st["changed"] += 1
                            for k in ("dmg:", "tk:x0", "tk:+", "tk:-", "fx:x", "tk<="):
                                if k in p1:
                                    kinds[k] += 1
                    obs = eng.select((agent if yi == 0 else oagent)(obs))
            except Exception:
                pass
            finally:
                eng.finish()
        per_deck[deck] = {"n": len(d_delta),
                          "changed": sum(1 for x in d_delta if x),
                          "mean_delta": sum(d_delta) / max(1, len(d_delta))}
        print("  %-24s %6d dec  changed %5.1f%%  +%.2f tok/decision"
              % (deck, len(d_delta), 100 * per_deck[deck]["changed"] / max(1, len(d_delta)),
                 per_deck[deck]["mean_delta"]), flush=True)

    def pct(v, q):
        v = sorted(v)
        return v[min(len(v) - 1, int(q * len(v)))] if v else 0

    n = len(base_tok) or 1
    print("\n%d decisions | changed %d (%.1f%%)" % (len(base_tok), st["changed"],
                                                    100 * st["changed"] / n))
    print("prompt tokens   mean %.1f -> %.1f   (+%.2f, +%.2f%%)"
          % (sum(base_tok) / n, sum(new_tok) / n,
             sum(delta) / n, 100 * sum(delta) / max(1, sum(base_tok))))
    print("                p90  %d -> %d      max %d -> %d"
          % (pct(base_tok, .9), pct(new_tok, .9), max(base_tok), max(new_tok)))
    nz = [d for d in delta if d]
    if nz:
        print("when it fires:  +%.2f tok mean, p90 +%d, max +%d"
              % (sum(nz) / len(nz), pct(nz, .9), max(nz)))
    print("\nfact frequency (decisions where the string appears):")
    for k, c in kinds.most_common():
        print("  %-8s %7d  %5.1f%%" % (k, c, 100 * c / n))
    if a.out:
        json.dump({"per_deck": per_deck, "base": base_tok, "delta": delta,
                   "kinds": dict(kinds)}, open(a.out, "w"))


if __name__ == "__main__":
    main()
