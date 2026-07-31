#!/usr/bin/env python3
"""Collect training records from states the LM'S OWN PLAY reaches, labelled by engine_v2.

The problem this fixes. Every existing record comes from engine_v2 self-play, so the model has
never seen the positions its own mistakes create -- engine_v2 diverges from the LM early and
never walks into them. Measured consequence: across the 36 collapsed decks the LM agrees with
engine_v2 on only 48.2% of decisions, and it is worse moving second (22.1%) than first (31.1%),
which is what distribution shift looks like.

Why not just reweight by action kind. Measured on 200,000 v39 records, `play` is already 25.9%
of labels -- the second most common. It is not under-covered. What is wrong is calibration:
`attach` and `end` are OFFERED on 20.1% and 9.7% of candidates but are the label only 8.0% and
3.9% of the time, and those are exactly the two the LM over-picks. The model falls back to
whatever is always on the menu. More `play` examples do not teach that; being shown its own
`play -> end` mistakes with the right answer does.

Records match tools/build_rerank.py's schema (state / candidates / chosen), so the output drops
straight into train_rerank alongside the existing pool.

engine_v2 is safe to use as the labeller here: tools/probe_engine_state.py measured its answer
to be a function of the observation alone (373/373 identical between an instance that had been
playing and a fresh one), so querying it on a state it never walked toward is well defined.
"""
import argparse
import collections
import gzip
import json
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "cg-lib")):
    if p not in sys.path:
        sys.path.insert(0, p)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--decks", default="", help="comma list; default = every deck")
    ap.add_argument("--model", required=True, help="hf:<dir> | qwen:<dir>")
    ap.add_argument("--games", type=int, default=40, help="per deck; seats alternate")
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep-agree", type=float, default=0.25,
                    help="fraction of decisions the LM ALREADY gets right to keep. Dropping "
                         "them all would concentrate the data on errors but also delete the "
                         "behaviour that currently works, so a slice is retained.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import library
    from cg.game import battle_start, battle_select, battle_finish
    from lm.actions import encode_option
    from lm.agent import make_lm_agent
    from lm.serialize import serialize_stateless
    from tools import rl_config
    from tools.mirror_match import make_agent

    fmt = dict(rl_config.PROMPT_FMT)
    decks = ([d.strip() for d in args.decks.split(",") if d.strip()]
             or sorted(library.list_decks()))
    tuning = json.load(open(os.path.join(ROOT, "agents", "tuning.json")))
    rng = random.Random(args.seed)
    st = collections.Counter()
    t0 = time.time()

    with gzip.open(args.out, "wt") as out:
        for di, deck in enumerate(decks):
            ids = [int(x) for x in open(library.deck_path(deck)) if x.strip()]
            prof = tuning.get(deck, {})
            lm_agent, _sc = make_agent(args.model, deck, ids, prof)
            ref = make_lm_agent(ids, prof, model=None)
            opp = make_lm_agent(ids, prof, model=None)
            for g in range(args.games):
                lm_seat = g % 2
                obs, _ = battle_start(ids, ids)
                if obs is None:
                    continue
                try:
                    for _ in range(4000):
                        cur = obs.get("current") or {}
                        if cur.get("result", -1) != -1 or obs.get("select") is None:
                            break
                        yi = cur.get("yourIndex", 0)
                        if yi != lm_seat:
                            obs = battle_select(opp(obs))
                            continue
                        opts = (obs.get("select") or {}).get("option") or []
                        pick_lm = lm_agent(obs)
                        pick_ref = ref(obs)
                        if len(opts) >= 2 and pick_lm and pick_ref \
                                and pick_ref[0] < len(opts) and pick_lm[0] < len(opts):
                            # dedupe by rendered text, as build_rerank does: identical option
                            # strings are one candidate, and the label must follow the survivor
                            seen, cands = {}, []
                            for i, o in enumerate(opts):
                                c = encode_option(o, obs)
                                if c not in seen:
                                    seen[c] = len(cands)
                                    cands.append(c)
                            lab = seen.get(encode_option(opts[pick_ref[0]], obs))
                            mine = seen.get(encode_option(opts[pick_lm[0]], obs))
                            if len(cands) >= 2 and lab is not None:
                                wrong = (lab != mine)
                                st["wrong" if wrong else "right"] += 1
                                if wrong or rng.random() < args.keep_agree:
                                    out.write(json.dumps({
                                        "state": serialize_stateless(
                                            obs, deck_ids=ids, deck_name=deck, **fmt),
                                        "candidates": cands, "chosen": lab,
                                        # the rendered menu is NOT deduped, so the decoder's
                                        # target is the RAW option index, not `chosen`
                                        "menu_index": pick_ref[0],
                                        "deck": deck, "seat": lm_seat,
                                        "lm_was_wrong": wrong}) + "\n")
                                    st["written"] += 1
                        obs = battle_select(pick_lm)
                finally:
                    battle_finish()
            print("[%2d/%d] %-24s written %6d  (LM wrong %.1f%%)  %.0fs"
                  % (di + 1, len(decks), deck, st["written"],
                     100.0 * st["wrong"] / max(1, st["wrong"] + st["right"]),
                     time.time() - t0), flush=True)

    print("\nwritten %d records | LM was wrong on %d of %d decisions (%.1f%%) | %.1f min"
          % (st["written"], st["wrong"], st["wrong"] + st["right"],
             100.0 * st["wrong"] / max(1, st["wrong"] + st["right"]),
             (time.time() - t0) / 60), flush=True)


if __name__ == "__main__":
    main()
