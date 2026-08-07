#!/usr/bin/env python3
"""Replay 4B self-play traces, branch its lowest-margin decisions, emit DPO preference pairs.

THE TRICK THAT MOVES STATES ACROSS MACHINES. The 4B plays mirror self-play on the GPU box and
its states are needed on the CPU box for playout measurement. Observation blobs are heavy;
what is cheap is (deck, seed, the exact pick sequence) -- and the engine is DETERMINISTIC given
a seed ([[engine-source-published]]), so feeding the recorded picks back into a fresh game
reconstructs every state bit-for-bit. The shuffle fingerprint in the trace header is checked
against the local build, and every traced decision is re-verified by its menu size: a single
divergence means the state is no longer the 4B's state, and the game is abandoned rather than
measured wrong ([[mirror-shuffle-mode]] carries the same guard for the screens).

WHAT BECOMES A PAIR. A decision qualifies when the model itself was UNSURE -- the margin
between its top choice and runner-up, recorded by the collector from inside the scorer. Both
are branched with rl_branch.branch_values: one determinization of the unseen pool shared
across both candidates per scenario, --playouts scenarios, engine_v2 continuations on both
sides (the known Q^engine compromise -- [[loss-gaps-priced-not-observed]] -- accepted for CPU
throughput). attach_label.label()'s four gates then decide whether the measured gap is real;
only decisions that pass become pairs:

    chosen   = the candidate the playouts prefer
    rejected = the other one
    model_was w|l  -- whether the policy already agreed. Both directions are kept (agreeing
                      pairs sharpen a correct margin; disagreeing pairs fix a wrong one) and
                      counted, so the trainer can rebalance if one side dominates.

Rows come out in the decoder SFT prompt format ({"prompt": ACT+state, "tw", "tl"}), so
dpo_teacher builds both completions with the same option_texts/label_b machinery the SFT
trainer uses -- one source of truth for what an answer token is.

    PYTHONPATH=cg-lib python3 tools/dpo_branch.py --traces /root/traces_r1.jsonl.gz \\
        --budget 8000 --workers 36 --out /root/ptcg/repo/data/rerank/dpo_r1.jsonl.gz
"""

import argparse
import collections
import gzip
import json
import multiprocessing as mp
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "cg-lib"), os.path.join(ROOT, "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

ACT = "[ACT]\n"          # must match valued_to_sft / the SFT pool


def _one_game(job):
    (deck, seed, picks, targets, playouts, wseed, fmt, so, fp_want) = job
    import library
    from lm.actions import encode_option
    from lm.agent import make_lm_agent
    from lm.serialize import serialize_stateless
    from attach_label import label
    from mirror_env import MirrorEngine, engine_fingerprint
    import rl_branch

    # The engine is initialised HERE, in the worker, and never in the parent. The pool uses a
    # spawn context for the same reason: a native library initialised before fork() leaves six
    # children sharing one engine's inherited state, and the crash it produces ("buffer full.
    # capacity:7", a C++ terminate) names none of that. qlabel_gen only ever touches the engine
    # inside its workers; price_targets never forks; this tool must hold both rules at once.
    global _ENG, _ROLL
    if "_ENG" not in globals() or _ENG is None:
        _ENG = MirrorEngine(so)
        fp = engine_fingerprint(_ENG, [int(x) for x in open(library.deck_path(
            sorted(library.list_decks())[0])) if x.strip()])
        if fp_want and fp != fp_want:
            raise SystemExit("worker fingerprint %s != trace %s" % (fp, fp_want))
    eng = _ENG
    if "_ROLL" not in globals():
        _ROLL = {}
    if deck not in _ROLL:
        ids = [int(x) for x in open(library.deck_path(deck)) if x.strip()]
        tuning = json.load(open(os.path.join(ROOT, "agents", "tuning.json")))
        prof = tuning.get(deck, {})
        # engine_v2 continuations for the playouts; the POLICY under test never runs here
        _ROLL[deck] = (ids, make_lm_agent(ids, prof, model=None),
                       make_lm_agent(ids, prof, model=None))
    ids, me, opp = _ROLL[deck]

    rng = random.Random(wseed)
    out, st = [], collections.Counter()
    want = {t: (margin, alt, nc) for t, margin, alt, nc in targets}
    obs = eng.start(ids, ids, seed, mirror=1)
    if obs is None:
        st["start_failed"] += 1
        return out, dict(st)
    t = 0
    try:  # noqa: SIM105 -- eng.finish() MUST run however this exits (see the finally)
        while t < len(picks):
            cur = obs.get("current") or {}
            if cur.get("result", -1) != -1 or obs.get("select") is None:
                break
            raw = (obs.get("select") or {}).get("option") or []
            pick = picks[t]
            if pick is None:
                break                                     # the original game forfeited here
            if t in want:
                margin, alt, nc_rec = want[t]
                # Sync check BEFORE branching: a replay that has drifted produces states the
                # 4B never saw, and a Q measured there labels nothing. Menu size is cheap and
                # catches drift immediately (any divergence changes what is on offer).
                if nc_rec is not None and nc_rec != len(raw):
                    st["desync"] += 1
                    break
                if not (isinstance(pick, list) and len(pick) == 1 and isinstance(alt, int)
                        and 0 <= pick[0] < len(raw) and 0 <= alt < len(raw)):
                    st["bad_target"] += 1
                else:
                    sels = [list(pick), [alt]]
                    per = [[], []]
                    # NOTE the shape of this block: whatever happens, control falls THROUGH to
                    # the shared eng.select(pick)/t+=1 at the bottom. An `eng.select + continue`
                    # inside the except -- qlabel_gen's pattern, which is correct in its plain
                    # for-loop -- double-applies the pick in this indexed loop (continue skips
                    # t += 1), and every decision after it is replayed against a shifted state.
                    best = means = None
                    labelable = True
                    try:
                        for _ in range(playouts):
                            q = rl_branch.branch_values(obs, ids, ids, 0, sels, me, opp,
                                                        n_playouts=1, rng=rng)
                            for i, v in enumerate(q):
                                if v is not None:
                                    per[i].append(v)
                        best, means = label(per, rng)
                    except rl_branch.DeterminizationError:
                        st["drop_determinize"] += 1       # mid-resolution state: not labelable
                        labelable = False
                    if best is None:
                        if labelable:
                            st["drop_neutral"] += 1
                    else:
                        iw_raw, il_raw = sels[best][0], sels[1 - best][0]
                        state = serialize_stateless(obs, deck_ids=ids, deck_name=deck, **fmt)
                        # PROMPT_FMT renders the menu DEDUPED (menu_dedup=True), so a raw obs
                        # index is a coordinate in the wrong space -- 20% land past the end and
                        # are at least visibly dropped; the rest silently point at whatever
                        # slid into that position. valued_to_sft named this exact failure ("a
                        # target index that points at the wrong option is invisible in
                        # training") and its fix is reused here: match the ACT through
                        # canon_key into the menu the prompt actually shows.
                        from lm.actions import encode_option as _enc
                        from lm.action_token import canon_key, slot_map_from_state
                        from valued_to_sft import menu_of
                        menu = menu_of(state)
                        if menu is None:
                            st["drop_no_menu"] += 1
                            obs = eng.select(pick)
                            if obs is None:
                                break
                            t += 1
                            continue
                        slots = slot_map_from_state(state)
                        mkeys = [canon_key(x, slots) for x in menu]

                        def _midx(raw_i):
                            want = canon_key(_enc(raw[raw_i], obs), slots)
                            return next((i for i, k in enumerate(mkeys) if k == want), None)
                        iw, il = _midx(iw_raw), _midx(il_raw)
                        if iw is None or il is None or iw == il:
                            st["drop_menu_match"] += 1
                            obs = eng.select(pick)
                            if obs is None:
                                break
                            t += 1
                            continue
                        out.append({
                            "prompt": ACT + state, "tw": str(iw), "tl": str(il),
                            "qw": round(means[best], 4), "ql": round(means[1 - best], 4),
                            "margin": margin, "model_was": "w" if best == 0 else "l",
                            "deck": deck, "seed": seed, "t": t,
                            "seat": (cur.get("yourIndex", 0)),
                        })
                        st["pair"] += 1
                        st["model_" + ("right" if best == 0 else "wrong")] += 1
            obs = eng.select(pick)
            if obs is None:
                break
            t += 1
    except Exception as e:                                # noqa: BLE001 -- one game, not the batch
        st["err_" + type(e).__name__] += 1
    finally:
        # mirror_env.play() closes every battle in a finally; a replay that does not leaks an
        # open instance per game into a buffer of capacity SEVEN, which is the crash above.
        eng.finish()
    if t < len(picks) and "desync" not in st:
        st["short_replay"] += 1
    return out, dict(st)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traces", required=True, help="comma-separated lm_mirror_log --trace-out files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget", type=int, default=8000, help="total branch points")
    ap.add_argument("--per-game", type=int, default=3)
    ap.add_argument("--margin-min", type=float, default=0.01,
                    help="skip margins below this. A margin of exactly 0 is two candidates the "
                         "scorer cannot tell apart AT ALL -- overwhelmingly two genuinely "
                         "equivalent moves the dedup could not prove equal -- and the playouts "
                         "agree (neutral), so they spend budget and return nothing. The user's "
                         "original DPO sketch excluded same-card flips for exactly this reason.")
    ap.add_argument("--playouts", type=int, default=16)
    ap.add_argument("--workers", type=int, default=36)
    ap.add_argument("--seed", type=int, default=11000)
    ap.add_argument("--mirror-so", default="")
    a = ap.parse_args()

    from mirror_env import DEFAULT_SO
    import rl_config

    so = a.mirror_so or DEFAULT_SO
    # NO engine in this (parent) process -- not even for the fingerprint. Each spawned worker
    # verifies the trace fingerprint against its own engine on first use instead.
    games, fp_remote = {}, None
    for path in [p for p in a.traces.split(",") if p]:
        with gzip.open(path, "rt") as f:
            for line in f:
                d = json.loads(line)
                if d.get("header"):
                    fp_remote = d.get("fp")
                    continue
                games[(d["deck"], d["seed"])] = d
    if not fp_remote:
        sys.exit("traces carry no fingerprint header -- refusing to replay blind")
    print("%d games loaded | trace fingerprint %s (workers will verify)"
          % (len(games), fp_remote), flush=True)

    # Lowest-margin decisions first, budgeted with per-deck fairness: interleave the decks'
    # own margin-sorted lists so one deck's cheap uncertainty cannot eat the whole budget
    # ([[narrow-dagger-overfits]] is the shape that guards against).
    per_deck = collections.defaultdict(list)
    for (deck, seed), g in games.items():
        cands = []
        for t, m in enumerate(g["meta"]):
            seat, margin, alt, nc = m[0], m[1], m[2], m[3]
            if margin is not None and alt is not None and margin >= a.margin_min:
                cands.append((margin, t, alt, nc))
        cands.sort()
        for margin, t, alt, nc in cands[:a.per_game]:
            per_deck[deck].append((margin, seed, t, alt, nc))
    for deck in per_deck:
        per_deck[deck].sort()
    chosen, i = [], 0
    while len(chosen) < a.budget:
        added = False
        for deck in sorted(per_deck):
            if i < len(per_deck[deck]):
                chosen.append((deck,) + per_deck[deck][i])
                added = True
                if len(chosen) >= a.budget:
                    break
        if not added:
            break
        i += 1
    print("selected %d branch points over %d decks (margin p50 %.3f)"
          % (len(chosen), len(per_deck),
             sorted(c[1] for c in chosen)[len(chosen) // 2] if chosen else 0.0), flush=True)

    by_game = collections.defaultdict(list)
    for deck, margin, seed, t, alt, nc in chosen:
        by_game[(deck, seed)].append((t, margin, alt, nc))
    fmt = dict(rl_config.PROMPT_FMT)
    jobs = [(deck, seed, games[(deck, seed)]["picks"], tgts, a.playouts,
             a.seed + 977 * j, fmt, so, fp_remote)
            for j, ((deck, seed), tgts) in enumerate(sorted(by_game.items()))]

    t0 = time.time()
    rows, agg = [], collections.Counter()
    with mp.get_context("spawn").Pool(min(a.workers, max(1, len(jobs)))) as p:
        for k, (out, st) in enumerate(p.imap_unordered(_one_game, jobs)):
            rows += out
            for kk, v in st.items():
                agg[kk] += v
            if (k + 1) % 200 == 0:
                print("  %d/%d games | %d pairs | %.0fs" % (k + 1, len(jobs), len(rows),
                                                            time.time() - t0), flush=True)

    with gzip.open(a.out, "wt") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("\nwrote %d pairs to %s in %.1f min" % (len(rows), a.out, (time.time() - t0) / 60))
    for k in sorted(agg):
        print("  %-18s %d" % (k, agg[k]))
    if agg.get("desync"):
        frac = agg["desync"] / max(1, len(jobs))
        print("DESYNC on %.1f%% of games%s" % (100 * frac,
              " -- STOP AND INVESTIGATE, the replay is not reproducing the collector's games"
              if frac > 0.02 else ""))
    if not rows:
        sys.exit("no pairs survived the gates")


if __name__ == "__main__":
    main()
