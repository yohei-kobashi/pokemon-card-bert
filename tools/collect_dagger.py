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
    # ---- seeded collection ------------------------------------------------------------------
    # 0 keeps the legacy unseeded engine. Non-zero drives the patched engine
    # (tools/build_engine_mirror.py) so a game is a function of its seed.
    #
    # SEED SPACE, kept disjoint on purpose. mirror_match's screen occupies 1..~65,600
    # (args.seed + crc32(deck)&0xFFFF); collection starts at 100,000; anchors sit at 2e9. An
    # overlap between screen and collection seeds would train the model on the very shuffles the
    # gate scores it on.
    #
    # Within one process the seed is base + deck_index*1000 + game, so the CALLER must space
    # bases by >= 100,000 per (round, pass, shard). Two shards sharing a base would replay
    # identical games -- harmless while the engine ignored seeds, silently a 1/3 collection once
    # it does not.
    ap.add_argument("--engine-seed-base", type=int, default=0)
    ap.add_argument("--mirror-so", default="")
    ap.add_argument("--mirror-shuffle", type=int, default=0,
                    help="1 = both seats get the SAME shuffle. Off by default: the two seats "
                         "already share a decklist here, and correlating their draws narrows "
                         "the state distribution DAgger exists to widen.")
    # A slice of games replayed on FIXED seeds every round. Same opening, same deck order, so
    # the LM's error rate on them is a paired measurement across rounds -- and it resolves in
    # decisions (tens of thousands) rather than games (tens), which is why it can see a round's
    # effect that the 40-game screen cannot. It measures AGREEMENT WITH engine_v2, not win rate,
    # so it is a leading indicator and not a substitute for the gate.
    # A FIXED panel of decks, played every round with seeds keyed by the DECK NAME. Both of
    # those matter. Keying the seed by the deck's INDEX in --decks made the anchor follow the
    # deck's position in a target list that is re-chosen every round, so the same deck drew
    # different games (dragapult: di=1 -> 101xxx one round, di=2 -> 202xxx the next); and taking
    # the panel from the targets meant a deck that left the tier stopped being measured, and the
    # anchor count moved with GAMES. None of that is comparable across rounds.
    ap.add_argument("--anchor-decks", default="", help="comma list; fixed across rounds")
    ap.add_argument("--anchor-games", type=int, default=0, help="per anchor deck, absolute")
    ap.add_argument("--anchor-frac", type=float, default=0.0,
                    help="legacy: carve this fraction out of --games on the TARGET decks. Not "
                         "comparable across rounds when the target list changes; prefer "
                         "--anchor-decks/--anchor-games.")
    ap.add_argument("--anchor-base", type=int, default=2_000_000_000)
    # Anchored games are HELD OUT by default: their rows are counted but not written. Writing
    # them would put the exact states -- with engine_v2's answer as the label -- into the corpus
    # the next round trains on, and the same seed replays the same opening, so next round's
    # anchor error rate would fall by memorisation rather than by generalisation. That turns the
    # measurement into a restatement of what was trained. The cost is anchor_frac of the round's
    # collected rows, which is why the fraction is small.
    ap.add_argument("--anchor-holdout", type=int, default=1)
    args = ap.parse_args()

    import library
    from lm.actions import encode_option
    from lm.action_token import dedup_options
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

    # Only ONE engine is loaded: importing cg.game would map the shipped libcg.so as well.
    per_game = {"anchor": [], "fresh": []}    # (wrong, total) per game
    per_deck = collections.defaultdict(lambda: [0, 0])   # deck -> [anchor wrong, anchor total]
    eng = None
    if args.engine_seed_base:
        from tools.mirror_env import DEFAULT_SO, MirrorEngine
        eng = MirrorEngine(args.mirror_so or DEFAULT_SO)
        battle_select, battle_finish = eng.select, eng.finish
        panel_n = len([d for d in args.anchor_decks.split(",") if d.strip()])
        print("[seeded] base %d | anchor panel %d decks x %d games from %d | same-shuffle %d"
              % (args.engine_seed_base, panel_n, args.anchor_games, args.anchor_base,
                 args.mirror_shuffle), flush=True)
    else:
        from cg.game import battle_finish, battle_select, battle_start  # noqa: F401

    import zlib

    def anchor_seed(deck, g):
        """Keyed by the deck NAME, so the panel is the same games whatever else the round does.
        crc32 & 0x1FFFFF gives 2,097,152 slots at a 1,000-game stride; 2e9 + that stays inside
        uint32, and a slot collision only means two decks share seeds -- different decklists, so
        different games anyway."""
        return args.anchor_base + (zlib.crc32(deck.encode()) & 0x1FFFFF) * 1000 + g

    panel = [d.strip() for d in args.anchor_decks.split(",") if d.strip()]
    work = collections.OrderedDict((d, [args.games, 0]) for d in decks)
    if eng and panel:
        for d in panel:
            work.setdefault(d, [0, 0])[1] = args.anchor_games
    elif eng and args.anchor_frac:
        na = int(round(args.games * args.anchor_frac))
        for d in work:
            work[d] = [args.games - na, na]

    with gzip.open(args.out, "wt") as out:
        for di, (deck, (n_fresh, n_anchor)) in enumerate(work.items()):
            ids = [int(x) for x in open(library.deck_path(deck)) if x.strip()]
            prof = tuning.get(deck, {})
            lm_agent, _sc = make_agent(args.model, deck, ids, prof)
            ref = make_lm_agent(ids, prof, model=None)
            opp = make_lm_agent(ids, prof, model=None)
            for g in range(n_anchor + n_fresh):
                lm_seat = g % 2
                is_anchor = g < n_anchor
                seed = (anchor_seed(deck, g) if is_anchor
                        else args.engine_seed_base + di * 1000 + (g - n_anchor))
                if eng:
                    obs = eng.start(ids, ids, seed, mirror=args.mirror_shuffle)
                else:
                    obs, _ = battle_start(ids, ids)
                if obs is None:
                    continue
                tag = "anchor" if is_anchor else "fresh"
                # Per-GAME keep-agree draws. One shared stream made an anchored game's kept rows
                # depend on how many draws the games before it had consumed, so changing the
                # fresh seeds silently re-selected rows inside the anchored -- supposedly fixed --
                # slice (162 rows vs 163 on the same games). Keyed by the game's own seed, an
                # anchor replays row for row.
                grng = random.Random((args.seed << 32) ^ seed ^ (di << 16) ^ g)
                gw = gn = 0
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
                            # Dedupe by the ACT, as build_rerank does -- not by rendered text.
                            # Two copies of a card in a shuffled pile, or two face-down prizes,
                            # are the same move written differently; keeping both puts one in the
                            # positive slot and its twin among the negatives, and no feature
                            # separates them. Measured on the reranker pool: 17.17% of records.
                            raw = [encode_option(o, obs) for o in opts]
                            cands, pos, keys = dedup_options(raw, obs)
                            idx = {keys[p]: n for n, p in enumerate(pos)}
                            lab = idx.get(keys[pick_ref[0]])
                            mine = idx.get(keys[pick_lm[0]])
                            if len(cands) >= 2 and lab is not None:
                                wrong = (lab != mine)
                                st["wrong" if wrong else "right"] += 1
                                st["%s_%s" % (tag, "wrong" if wrong else "right")] += 1
                                gw += wrong
                                gn += 1
                                if is_anchor and args.anchor_holdout:
                                    pass          # measured above, deliberately not written
                                elif wrong or grng.random() < args.keep_agree:
                                    out.write(json.dumps({
                                        "state": serialize_stateless(
                                            obs, deck_ids=ids, deck_name=deck, **fmt),
                                        "candidates": cands, "chosen": lab,
                                        # STALE under v40. `fmt` comes from rl_config.PROMPT_FMT,
                                        # which sets menu_dedup=True, so the rendered menu IS the
                                        # deduped list and the decoder's target is `chosen`.
                                        # This raw option index is kept only for diagnostics;
                                        # tools/rerank_to_sft.py derives the target from the
                                        # rendered menu and checks the identity per record.
                                        "menu_index": pick_ref[0],
                                        # what the LM played INSTEAD. Without it the pool says
                                        # only which move was missed, so neither the confusion
                                        # matrix nor a two-branch value comparison between the
                                        # two moves can be built after the fact.
                                        "lm_chosen": mine, "lm_menu_index": pick_lm[0],
                                        "deck": deck, "seat": lm_seat,
                                        # which game produced this row, so a record can be traced
                                        # back to a replayable game and anchored rows can be
                                        # compared like-for-like across rounds.
                                        "seed": seed, "anchor": is_anchor,
                                        "lm_was_wrong": wrong}) + "\n")
                                    st["written"] += 1
                        obs = battle_select(pick_lm)
                finally:
                    battle_finish()
                if gn:
                    per_game[tag].append((gw, gn))
                    if is_anchor:
                        per_deck[deck][0] += gw
                        per_deck[deck][1] += gn
            print("[%2d/%d] %-24s written %6d  (LM wrong %.1f%%)  %.0fs"
                  % (di + 1, len(work), deck, st["written"],
                     100.0 * st["wrong"] / max(1, st["wrong"] + st["right"]),
                     time.time() - t0), flush=True)

    print("\nwritten %d records | LM was wrong on %d of %d decisions (%.1f%%) | %.1f min"
          % (st["written"], st["wrong"], st["wrong"] + st["right"],
             100.0 * st["wrong"] / max(1, st["wrong"] + st["right"]),
             (time.time() - t0) / 60), flush=True)
    # The line to track across rounds. Anchored games replay FIXED seeds, so this is a paired
    # error rate on the same openings -- it resolves in decisions, not games. Binomial SE only;
    # decisions within a game are correlated, so treat it as a leading indicator, not a gate.
    for tag in ("anchor", "fresh"):
        n = st[tag + "_wrong"] + st[tag + "_right"]
        if not n:
            continue
        p = st[tag + "_wrong"] / n
        se_binom = (p * (1 - p) / n) ** 0.5
        # Decisions inside one game are NOT independent -- they share a deal and a line of play --
        # so the binomial SE is a lower bound. The honest figure is the spread of the per-GAME
        # error rate: SE = sd(rate) / sqrt(games). Report both; their ratio is the design effect.
        rates = [w / m for w, m in per_game[tag] if m]
        se = se_binom
        if len(rates) > 1:
            import statistics as _s
            se = _s.stdev(rates) / len(rates) ** 0.5
        note = " (held out)" if tag == "anchor" and args.anchor_holdout else ""
        print("[%s] LM wrong %.2f%% +- %.2f (binomial +- %.2f, %d games / %d decisions)%s"
              % (tag, 100 * p, 100 * se, 100 * se_binom, len(rates), n, note), flush=True)
    # Per deck too: the panel is fixed, so these line up round to round and can be compared
    # PAIRED -- which is the whole point, and what an overall mean would throw away.
    for deck in sorted(per_deck):
        w, m = per_deck[deck]
        if m:
            print("[anchor-deck] %-24s %.2f%%  (%d decisions)" % (deck, 100 * w / m, m), flush=True)


if __name__ == "__main__":
    main()
