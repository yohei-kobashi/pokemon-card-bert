#!/usr/bin/env python3
"""Playout-measured Q labels for the (deck, kind) cells the loss analysis flagged.

`tools/attach_label.py` does exactly this for ONE decision kind. It is not written to be
general -- its sampling weights are keyed to attach's measured headroom per (n_targets, prizes)
cell, and its records carry attach-specific fields -- so this is a sibling rather than a flag
on it. The STATISTICS are imported from it, not copied: `label` (permutation null, split-sample
argmax, sign check, unique-argmax check) and `permutation_null` are the same four gates, and
having two copies of those drift apart is how a subtle labelling bug would survive review.

WHAT IS DIFFERENT.

  targeting   The cells come from tools/diag_lm_losses.py --targets, i.e. from a controlled
              within-game contrast on mirror games (same deck, same shuffle, same policy, one
              seat won), optionally re-weighted by tools/price_targets.py once the observed
              gaps have been priced. Budget follows `share`, so mega_lucario_tr/evolve at 18.5%
              of the measured gap gets 18.5% of the branch points instead of the 1/11 a
              uniform sweep would give it.
  kinds       Any action kind, not just attach. A decision qualifies when a TARGETED kind is
              on the menu; the branch set is that kind's candidates plus the pilot's own pick
              plus a sample of the rest, so "the targeted kind loses here" is learnable too.
  no cell     attach's (k, prizes) headroom table does not describe evolve or retreat, and
  weights     inventing one per kind would be a guess presented as a measurement. Sampling is
              uniform within a cell; the targeting itself is the prior.

THE PILOT IS engine_v2 BY DEFAULT, and that is a deliberate compromise with a known cost. The
gaps were observed in LM-piloted games, so the states that matter are LM-reached, and engine
states are not the same distribution. attach_label makes the same trade and states why: the
question is whether the DECISION holds signal, not whether one pilot walks into it -- and the
resulting labels did help the decoder (9.2% of its mix). What it buys is throughput: an engine
playout is ~0.27 s of pure CPU and parallelises across instance1's 61 effective cores, while an
LM-piloted collection is GPU-serial. Pass --pilot qwen:<dir> to pay for on-policy states.

    PYTHONPATH=cg-lib python3 tools/qlabel_gen.py --targets evaluations/lm_targets.json \\
        --per-deck 400 --workers 40 --out data/rerank/qlabel_r1.jsonl.gz
"""

import argparse
import collections
import gzip
import json
import multiprocessing as mp
import os
import random
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "cg-lib"), os.path.join(ROOT, "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def kind_of(t):
    m = re.match(r"([a-z_]+)", t or "")
    return m.group(1) if m else "?"


def _one_deck(job):
    # Backstop: whatever this deck hit, the other ten decks' records are still worth keeping.
    # imap_unordered re-raises a worker exception in the PARENT, so an unguarded failure here
    # aborts the whole batch and the loop writes an empty file.
    try:
        return _run_deck(job)
    except Exception as e:                                          # noqa: BLE001
        import traceback
        traceback.print_exc()
        return job[0], [], {"worker_error": 1, "err_" + type(e).__name__: 1}, 0.0


def _run_deck(job):
    (deck, kinds, games, playouts, seed, target, fmt, budget, other_k, max_branch,
     pilot_spec, seat) = job
    import library
    from cg.game import battle_start, battle_select, battle_finish
    from lm.actions import encode_option
    from lm.action_token import dedup_options
    from lm.agent import make_lm_agent
    from lm.serialize import serialize_stateless
    from attach_label import label
    import rl_branch

    rng = random.Random(seed)
    ids = [int(x) for x in open(library.deck_path(deck)) if x.strip()]
    tuning = json.load(open(os.path.join(ROOT, "agents", "tuning.json")))
    prof = tuning.get(deck, {})
    # The ROLLOUT pilot is always engine_v2: a playout that consulted the LM would cost a GPU
    # call per ply and there are 16 x branches of them per decision.
    me = make_lm_agent(ids, prof, model=None)
    opp = make_lm_agent(ids, prof, model=None)
    pilot = me
    if pilot_spec:
        from mirror_match import make_agent
        pilot, _sc = make_agent(pilot_spec, deck, ids, prof)

    out, st = [], collections.Counter()
    t0 = time.time()
    want = set(kinds)
    for _g in range(games):
        if len(out) >= target or time.time() - t0 > budget:
            break
        obs, _ = battle_start(ids, ids)
        if obs is None:
            continue
        # Whoever is asked to act first IS the first player, so the seat filter can be resolved
        # from the opening observation without any engine-internal field. Needed because the
        # Alakazam family's failure is seat-conditioned -- 34-6 and 32-8 for the first player in
        # the mirror log -- and labels collected in the seat it already wins from cannot fix it.
        first_actor = None
        try:
            for _ in range(4000):
                cur = obs.get("current") or {}
                if cur.get("result", -1) != -1 or obs.get("select") is None:
                    break
                yi = cur.get("yourIndex", 0)
                if first_actor is None:
                    first_actor = yi
                    we_go_first = (first_actor == 0)
                    if (seat == "first" and not we_go_first) or \
                            (seat == "second" and we_go_first):
                        st["skip_seat"] += 1
                        break
                opts = (obs.get("select") or {}).get("option") or []
                pick = (pilot if yi == 0 else opp)(obs)
                if yi != 0 or len(opts) < 2 or not pick or pick[0] >= len(opts) \
                        or len(out) >= target or time.time() - t0 > budget:
                    obs = battle_select(pick)
                    continue
                raw = [encode_option(o, obs) for o in opts]
                cands, pos, keys = dedup_options(raw, obs)
                tgt = [i for i, t in enumerate(cands) if kind_of(t) in want]
                if not tgt or len(cands) < 2:
                    obs = battle_select(pick)
                    continue
                st["seen"] += 1
                lab = {keys[p]: n for n, p in enumerate(pos)}.get(keys[pick[0]])
                # The pilot's own pick must be in the branch set or "the targeted kind beats
                # what we actually do" is unanswerable -- the comparison would be against a
                # sample of alternatives that may not include the status quo.
                oth = [i for i in range(len(cands)) if i not in tgt]
                if lab is not None and lab not in tgt and lab in oth:
                    oth.remove(lab)
                    extra = [lab] + (oth if len(oth) <= other_k - 1
                                     else rng.sample(oth, max(0, other_k - 1)))
                else:
                    extra = oth if len(oth) <= other_k else rng.sample(oth, other_k)
                branch = tgt[:max_branch] + extra
                if len(branch) < 2:
                    obs = battle_select(pick)
                    continue
                sels = [[pos[i]] for i in branch]
                per = [[] for _ in sels]
                # attach_label never needed this guard: an attach decision is always taken at a
                # clean turn boundary, where every card is either visible or in the unseen pool.
                # Targeting arbitrary kinds reaches MID-RESOLUTION states -- a sub-select inside
                # a search, a card being moved -- where the visible-card accounting is transient
                # and unseen_multisets legitimately refuses to determinize. That is one decision
                # we cannot label, not a broken batch, so it is counted and skipped. Before this
                # guard the exception propagated out of imap_unordered and discarded every
                # record from all 11 decks.
                try:
                    for _s in range(playouts):
                        q = rl_branch.branch_values(obs, ids, ids, 0, sels, me, opp,
                                                    n_playouts=1, rng=rng)
                        for i, v in enumerate(q):
                            if v is not None:
                                per[i].append(v)
                except rl_branch.DeterminizationError:
                    st["drop_determinize"] += 1
                    obs = battle_select(pick)
                    continue
                best, means = label(per, rng)
                if best is None:
                    st["drop_neutral"] += 1
                else:
                    qv = [None] * len(cands)
                    for i, c in enumerate(branch):
                        qv[c] = means[i]
                    won = kind_of(cands[branch[best]])
                    out.append({
                        "state": serialize_stateless(obs, deck_ids=ids, deck_name=deck, **fmt),
                        "candidates": cands, "chosen": branch[best], "qvals": qv,
                        "engine_chosen": lab, "kind": "qlabel", "won_kind": won,
                        "target_kinds": sorted(want), "deck": deck, "opp": deck,
                        "valued": True})
                    st["kept"] += 1
                    st["won_" + won] += 1
                    if lab in branch:
                        st["pilot_branched"] += 1
                        st["pilot_agreed"] += (branch[best] == lab)
                obs = battle_select(pick)
        finally:
            battle_finish()
    return deck, out, dict(st), time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--games", type=int, default=600, help="cap per deck")
    ap.add_argument("--per-deck", type=int, default=400, help="record target per deck at share 1")
    ap.add_argument("--playouts", type=int, default=16)
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--other-k", type=int, default=3, help="non-target candidates branched")
    ap.add_argument("--max-branch", type=int, default=5, help="cap on target-kind candidates")
    ap.add_argument("--deck-seconds", type=float, default=2400)
    ap.add_argument("--seed", type=int, default=9000)
    ap.add_argument("--pilot", default="", help="'' = engine_v2 (default) | qwen:<dir> | hf:<dir>")
    ap.add_argument("--seat", choices=("any", "first", "second", "cell"), default="cell",
                    help="label only from this seat; 'cell' reads each cell's own `seat` field")
    a = ap.parse_args()

    import rl_config
    fmt = dict(rl_config.PROMPT_FMT)
    cells = json.load(open(a.targets))["cells"]
    by_deck = collections.defaultdict(lambda: {"kinds": set(), "share": 0.0, "seat": "any"})
    for c in cells:
        by_deck[c["deck"]]["kinds"].add(c["kind"])
        by_deck[c["deck"]]["share"] += c.get("share", 0.0)
        # A cell may pin the seat. Cells of the same deck that disagree fall back to "any"
        # rather than letting whichever was read last decide.
        s, have = c.get("seat", "any"), by_deck[c["deck"]]["seat"]
        by_deck[c["deck"]]["seat"] = s if have in ("any", s) else "any"
    if not by_deck:
        sys.exit("no cells in %s" % a.targets)

    tot = sum(v["share"] for v in by_deck.values()) or 1.0
    jobs = []
    for i, (deck, v) in enumerate(sorted(by_deck.items())):
        share = v["share"] / tot
        # Budget follows the measured gap. A floor keeps a small-gap deck from vanishing --
        # a deck with zero records cannot be shown to have been fixed OR not fixed.
        tgt = max(60, int(round(a.per_deck * len(by_deck) * share)))
        seat = a.seat if a.seat != "cell" else v["seat"]
        # A seat filter throws away ~half the games, so the game cap has to grow with it or the
        # deck silently returns half its target and looks merely slow.
        games = a.games if seat == "any" else a.games * 2
        jobs.append((deck, sorted(v["kinds"]), games, a.playouts, a.seed + 137 * i, tgt,
                     fmt, a.deck_seconds, a.other_k, a.max_branch, a.pilot, seat))
        print("  %-20s kinds %-28s share %5.1f%% seat %-6s -> target %d records"
              % (deck, ",".join(sorted(v["kinds"]))[:28], 100 * share, seat, tgt), flush=True)

    t0 = time.time()
    agg, rows = collections.Counter(), []
    with mp.Pool(min(a.workers, len(jobs))) as p:
        for deck, out, st, dt in p.imap_unordered(_one_deck, jobs):
            rows += out
            for k, v in st.items():
                agg[k] += v
            print("  %-20s kept %4d / seen %5d  %4.0fs  (total %d)"
                  % (deck, st.get("kept", 0), st.get("seen", 0), dt, len(rows)), flush=True)

    with gzip.open(a.out, "wt") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print("\nwrote %d records to %s in %.1f min" % (len(rows), a.out, (time.time() - t0) / 60))
    seen, kept = agg.get("seen", 0), agg.get("kept", 0)
    print("  branch points seen        %d" % seen)
    print("  dropped as value-neutral  %d (%.1f%% of branched)"
          % (agg.get("drop_neutral", 0), 100.0 * agg.get("drop_neutral", 0) / max(1, seen)))
    if agg.get("skip_seat"):
        print("  games skipped, wrong seat  %d" % agg["skip_seat"])
    if agg.get("drop_determinize"):
        print("  dropped, not determinizable %d (%.1f%% of branched)"
              % (agg["drop_determinize"], 100.0 * agg["drop_determinize"] / max(1, seen)))
    for k in sorted(k for k in agg if k.startswith("err_")):
        print("  WORKER ERROR %-22s %d deck(s)" % (k[4:], agg[k]))
    if agg.get("pilot_branched"):
        print("  label == the pilot's pick %d (%.1f%% of the %d where its pick was branched)"
              % (agg["pilot_agreed"], 100.0 * agg["pilot_agreed"] / agg["pilot_branched"],
                 agg["pilot_branched"]))
    for k in sorted(k for k in agg if k.startswith("won_")):
        print("  playouts chose %-10s %d (%.1f%% of kept)"
              % (k[4:], agg[k], 100.0 * agg[k] / max(1, kept)))
    if not rows:
        sys.exit("no records: every branch point was value-neutral, or none were reached")


if __name__ == "__main__":
    main()
