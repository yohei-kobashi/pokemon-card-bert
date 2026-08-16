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


# Defaults only. The real values arrive per job (see main): the pool is a SPAWN context, so a
# module global assigned in main() never reaches a worker.
_EMIT_RULE_W = False
_RULE_EXCLUDE = frozenset()


def _one_game(job):
    (deck, seed, picks, targets, playouts, wseed, fmt, so, fp_want, rule_cfg, doc_cfg) = job
    doc_rules, doc_cap, doc_deck = doc_cfg
    doc_left = doc_cap if doc_rules else 0
    global _EMIT_RULE_W, _RULE_EXCLUDE
    _EMIT_RULE_W, _RULE_EXCLUDE = rule_cfg
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
    # A trace names one deck (same-deck self-play) or two (protagonist vs opponent). Seat 0
    # holds d0, seat 1 holds d1; the engine has always taken the two decks separately and is
    # deterministic either way (verified: with different decklists, same seed, each seat still
    # gets a permutation of ITS OWN list and the order repeats across runs).
    d0, d1 = (deck, deck) if isinstance(deck, str) else (deck[0], deck[1])
    tuning = json.load(open(os.path.join(ROOT, "agents", "tuning.json")))
    for d in (d0, d1):
        if d not in _ROLL:
            dids = [int(x) for x in open(library.deck_path(d)) if x.strip()]
            # engine_v2 continuations for the playouts; the POLICY under test never runs here
            _ROLL[d] = (dids, make_lm_agent(dids, tuning.get(d, {}), model=None))
    IDS = (_ROLL[d0][0], _ROLL[d1][0])
    AGENTS = (_ROLL[d0][1], _ROLL[d1][1])

    rng = random.Random(wseed)
    out, st = [], collections.Counter()
    want = {t: (margin, alt, nc) for t, margin, alt, nc in targets}
    obs = eng.start(IDS[0], IDS[1], seed, mirror=1)
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
            # Doctrine seeding: inject a synthetic target so the branch flows through the
            # SAME playout/label/emit path as a margin-selected one (margin recorded as 0.0,
            # nc as None -- the sync check skips None). Only the seeded deck's own decisions,
            # only when the model did NOT already conform, only doc_cap times a game.
            if doc_left > 0 and t not in want and isinstance(pick, list) and len(pick) == 1:
                _yi0 = cur.get("yourIndex", 0)
                if doc_deck is None or (d0, d1)[_yi0] == doc_deck:
                    try:
                        import dusk_plan
                        _live = dusk_plan.opportunities(obs, _yi0)
                    except Exception:                      # noqa: BLE001
                        _live = {}
                    for _rn in doc_rules:
                        _hit = _live.get(_rn)
                        if not _hit or not _hit[0]:
                            continue
                        if pick[0] in _hit[0]:
                            continue                       # the model already conforms
                        _nom = sorted(_hit[0])[0]
                        if 0 <= _nom < len(raw):
                            want[t] = (0.0, _nom, None)
                            doc_left -= 1
                            st["doctrine_" + _rn] += 1
                            break
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
                    # PERSPECTIVE. branch_values' `pilot_i` is the ABSOLUTE player index whose
                    # win scores +1 (rl_branch._playout: `1 if r == pilot_i else -1`), and
                    # `my_deck` is the ACTING player's list. This used to pass 0 and 0 always,
                    # so every branch point where the 4B was moving as player 1 -- 48% of them
                    # -- was labelled with the OPPONENT's preference: chosen and rejected
                    # swapped. It matched the gate exactly (seat1 -0.53pt at 25% such pairs,
                    # -6.10pt at 46%), i.e. "seat-fair selection made seat 1 worse" was this
                    # bug being fed more of its own poison, not a data-quantity effect.
                    yi = cur.get("yourIndex", 0)
                    try:
                        for _ in range(playouts):
                            q = rl_branch.branch_values(obs, IDS[yi], IDS[1 - yi], yi, sels,
                                                        AGENTS[yi], AGENTS[1 - yi],
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
                        # The prompt must describe the deck the ACTOR is holding. With two
                        # different lists in play, rendering seat 1's board against seat 0's
                        # DECK[...] would teach the model a list it is not piloting.
                        state = serialize_stateless(obs, deck_ids=IDS[yi],
                                                    deck_name=(d0, d1)[yi], **fmt)
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
                        rww = rwl = 0.0
                        if _EMIT_RULE_W:
                            try:
                                import dusk_plan
                                for _rn, (_good, _sc) in dusk_plan.opportunities(obs, yi).items():
                                    if _rn in _RULE_EXCLUDE:
                                        continue
                                    _wt = dusk_plan.RULES[_rn][1]
                                    if iw_raw in _good:
                                        rww += _wt
                                    if il_raw in _good:
                                        rwl += _wt
                            except Exception:                  # noqa: BLE001
                                pass                           # rules must never sink a pair
                        out.append({
                            "prompt": ACT + state, "tw": str(iw), "tl": str(il),
                            # The candidates as the MENU shows them -- what a cross-encoder
                            # scores. tw/tl are the 4B's index-token coordinates; these are the
                            # same two candidates in the reranker's coordinates.
                            "cw": menu[iw], "cl": menu[il],
                            "rww": round(rww, 2), "rwl": round(rwl, 2),
                            "qw": round(means[best], 4), "ql": round(means[1 - best], 4),
                            "margin": margin, "model_was": "w" if best == 0 else "l",
                            # `deck` is the deck the 4B is PILOTING at this decision, `opp`
                            # what it faces -- not the matchup label, so a mix can be weighted
                            # by either without re-deriving it from the seat.
                            "deck": (d0, d1)[yi], "opp": (d0, d1)[1 - yi],
                            "seed": seed, "t": t, "seat": yi,
                            # How much evidence is behind qw/ql. The trainer converts
                            # (qw - ql, pl) into a per-pair cDPO epsilon; without pl it would
                            # have to assume a playout count and mis-weight every label.
                            "pl": playouts,
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
    ap.add_argument("--no-seat-fair", dest="seat_fair", action="store_false",
                    help="spend the budget by margin alone, which gives the first seat ~2/3 "
                         "of it (see the selection block). Kept only to reproduce rounds 1-2.")
    ap.add_argument("--margin-min", type=float, default=0.01,
                    help="skip margins below this. A margin of exactly 0 is two candidates the "
                         "scorer cannot tell apart AT ALL -- overwhelmingly two genuinely "
                         "equivalent moves the dedup could not prove equal -- and the playouts "
                         "agree (neutral), so they spend budget and return nothing. The user's "
                         "original DPO sketch excluded same-card flips for exactly this reason.")
    ap.add_argument("--fmt", default="prompt", choices=("prompt", "dusk"),
                    help="'dusk' renders prompts in the single-deck no-DECK[] format for the "
                         "DeBERTa mirror-RL loop; 'prompt' is instance2's cross-deck default")
    ap.add_argument("--rule-weights", action="store_true",
                    help="also emit each candidate's summed dusk_plan rule weight (rww/rwl), "
                         "for blending rule conformance into the reward downstream")
    ap.add_argument("--rule-exclude", default="spread_aim,energy_line,energy_focus,recon",
                    help="rules NOT counted by --rule-weights: the deferred set that "
                         "make_plan_rule executes at inference, which the model never decides")
    ap.add_argument("--playouts", type=int, default=16)
    # DOCTRINE SEEDING (the reward-side half of the guide work, 2026-08-16). Margin-based
    # selection only ever branches where the MODEL is unsure, and the alternative is the
    # model's own runner-up -- a decision the model is confidently wrong about (never firing
    # the spike, never taking the Candy line) is never branched, so the playouts never get to
    # price the doctrine at all. These rules failed as inference-time CONSTRAINTS (crispin
    # family 3x negative, third_loak -1.67): here they only propose the comparison, and the
    # playout Q decides which side becomes the preferred half of the pair.
    ap.add_argument("--doctrine-rules", default="",
                    help="comma-separated dusk_plan rules used as branch SEEDS: at replayed "
                         "decisions where the rule nominates a move the model did not take, "
                         "branch (model move vs rule move) regardless of model margin")
    ap.add_argument("--doctrine-per-game", type=int, default=2)
    ap.add_argument("--doctrine-games", type=int, default=240,
                    help="games WITHOUT margin targets (e.g. engine-piloted traces, which "
                         "record no model margins) admitted for doctrine seeding alone")
    ap.add_argument("--workers", type=int, default=36)
    ap.add_argument("--only-deck", default="",
                    help="branch ONLY decisions made by this deck's side (per-deck opponent "
                         "adapters: the other side is dusknoir and must not be trained here)")
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
                # A cross-deck trace names deck0/deck1; a same-deck one names deck. The key
                # keeps the pair so the replay can hand each seat its own list.
                key = ((d["deck0"], d["deck1"]) if "deck0" in d
                       else (d["deck"], d["deck"]))
                games[(key, d["seed"])] = d
    if not fp_remote:
        sys.exit("traces carry no fingerprint header -- refusing to replay blind")
    print("%d games loaded | trace fingerprint %s (workers will verify)"
          % (len(games), fp_remote), flush=True)

    # Lowest-margin decisions first, budgeted with per-deck fairness: interleave the decks'
    # own margin-sorted lists so one deck's cheap uncertainty cannot eat the whole budget
    # ([[narrow-dagger-overfits]] is the shape that guards against).
    #
    # THE BUCKET IS (deck, SEAT), not deck. Measured on round 2's traces: the decisions
    # themselves split 51.9/48.1 across seats, but the policy is systematically LESS certain
    # when it moves first (margin p50 11.50 at seat0 vs 15.75 at seat1), so a pure
    # lowest-margin cut hands seat0 67.5% of the budget, and the label gate -- which drops a
    # branch point when the playouts cannot separate the two candidates, i.e. when the
    # position is already decided -- widens it again to 75/25 in the pairs. Round 1's entire
    # gate gain was seat0 (+4.46pt) with seat1 flat (-0.53pt): the second seat was not being
    # trained. The live ladder alternates seats, so that is half of every game left on the
    # table. Splitting the per-game cut and the interleave by seat costs nothing and makes
    # the budget spend the same on both.
    seats = (0, 1) if a.seat_fair else (0,)
    per_game_seat = max(1, a.per_game // len(seats))
    per_deck = collections.defaultdict(list)
    for (deck, seed), g in games.items():
        cands = collections.defaultdict(list)
        for t, m in enumerate(g["meta"]):
            seat, margin, alt, nc = m[0], m[1], m[2], m[3]
            if margin is not None and alt is not None and margin >= a.margin_min:
                # --only-deck spends the whole budget on ONE side's decisions. A trace of
                # X vs dusknoir carries both, and training X's adapter on dusknoir's moves
                # would teach X to pilot a deck it never holds -- half the budget, spent
                # backwards. `deck` here is the (deck0, deck1) pair and the seat indexes it.
                if a.only_deck and deck[seat] != a.only_deck:
                    continue
                cands[seat if a.seat_fair else 0].append((margin, t, alt, nc))
        for seat, cl in cands.items():
            cl.sort()
            for margin, t, alt, nc in cl[:per_game_seat]:
                per_deck[(deck, seat)].append((margin, seed, t, alt, nc))
    for k in per_deck:
        per_deck[k].sort()
    chosen, picked, i = [], collections.Counter(), 0
    while len(chosen) < a.budget:
        added = False
        for k in sorted(per_deck):
            if i < len(per_deck[k]):
                chosen.append((k[0],) + per_deck[k][i])
                picked[k[1]] += 1
                added = True
                if len(chosen) >= a.budget:
                    break
        if not added:
            break
        i += 1
    print("selected %d branch points over %d decks (margin p50 %.3f) | seat split %s"
          % (len(chosen), len({k[0] for k in per_deck}),
             sorted(c[1] for c in chosen)[len(chosen) // 2] if chosen else 0.0,
             " ".join("s%d %d" % (s, n) for s, n in sorted(picked.items()))), flush=True)

    by_game = collections.defaultdict(list)
    for deck, margin, seed, t, alt, nc in chosen:
        by_game[(deck, seed)].append((t, margin, alt, nc))
    # Margin selection only jobs games where the MODEL recorded uncertainty -- an
    # engine-piloted trace records none and would never reach the doctrine seeder.
    # Admit a bounded number of target-less games so their replays can still be seeded.
    if any(x for x in a.doctrine_rules.split(",") if x) and a.doctrine_games > 0:
        extra = [k for k in games if k not in by_game]
        random.Random(a.seed).shuffle(extra)
        for k in extra[:a.doctrine_games]:
            by_game[k] = []
        if extra:
            print("doctrine: admitted %d target-less games (of %d) for seeding"
                  % (min(len(extra), a.doctrine_games), len(extra)), flush=True)
    fmt = dict(rl_config.DUSK_FMT if a.fmt == "dusk" else rl_config.PROMPT_FMT)
    # THE CONFIG TRAVELS IN THE JOB TUPLE. It used to be assigned to module globals here, with
    # a comment claiming workers inherit it "by fork" -- but the pool below is a SPAWN context,
    # so every worker re-imports this module and gets the module-level defaults
    # (_EMIT_RULE_W = False). The effect was silent and total: `--rule-weights` emitted rww=rwl=0
    # on every pair of every mirror-RL round, so the rule-conformance third of the reward was
    # never in the reward at all, and the downstream beta-blend degenerated into 30% label
    # smoothing toward uniform.
    rule_cfg = (bool(a.rule_weights), frozenset(x for x in a.rule_exclude.split(",") if x))
    doc_cfg = (tuple(x for x in a.doctrine_rules.split(",") if x),
               a.doctrine_per_game, a.only_deck or None)
    jobs = [(deck, seed, games[(deck, seed)]["picks"], tgts, a.playouts,
             a.seed + 977 * j, fmt, so, fp_remote, rule_cfg, doc_cfg)
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
