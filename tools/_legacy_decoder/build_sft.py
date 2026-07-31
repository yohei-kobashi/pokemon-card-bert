"""Transform gen_selfplay logs -> SFT samples. Runs NO battles: pure file transform
on the structured JSONL.gz that gen_selfplay produced (see docs/ml_agent_plan.md §5,§6).

For each game it reconstructs the WINNER's append-only running context (the exact form
the live agent feeds, via lm.serialize.EpisodeSerializer) and, at each winner MAIN
selection, emits DUAL-MODE samples (plan §2):
  - mode "act"    : prompt -> action            (what inference uses; no reasoning)
  - mode "reason" : prompt -> reasoning + action (hindsight training signal)
The reasoning is COMPACT and EVENT-ANCHORED (plan §1): future winner-step logs up to
the next ATTACK/RESULT event (capped), then a one-line outcome. Prompt carries a mode
tag so the model knows which target to produce.

Usage:
    python tools/build_sft.py --tag _smoke_sft
    python tools/build_sft.py --tag <run> --window 6 --out data/sft
"""
import argparse
import glob
import gzip
import json
import math
import os
import random
import shutil
import sys
import time

# this file lives at tools/_legacy_decoder/, i.e. TWO levels below the repo root -- the
# two-dirname version resolved to tools/ and made it look for data/ under tools/.
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (ROOT, os.path.join(ROOT, "cg-lib"), os.path.join(ROOT, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

from lm.serialize import serialize_stateless, render_logs, multipick_substate, STOP  # noqa: E402

import rl_config  # noqa: E402


def _ser_cur(obs, deck_ids=None, deck_name=None):
    """serialize_stateless in the CURRENT prompt format.

    Bare serialize_stateless() renders the legacy full-rules glossary with no DECK[] and no
    `ID ME` -- a ~838-token prompt for models trained on ~245. rl_config.PROMPT_FMT is the single
    source of truth and the deck NAME is not optional (it renders `ID ME d_x a_y`)."""
    return serialize_stateless(obs, deck_ids=deck_ids, deck_name=deck_name,
                               **dict(rl_config.PROMPT_FMT))
from lm.actions import encode_option                       # noqa: E402
from eval_state import evaluate, WIN                        # noqa: E402
from cg.api import to_observation_class                     # noqa: E402


_SCORER = None      # set by main() for --adopt value; None == handcrafted evaluate()
_TURN_BOUNDARY = 0.0
"""Correction added to the delta when the move handed control to the opponent.

The learned scorer has `to_move` as a feature, so it correctly prices "it is now the
opponent's turn" -- which means EVERY turn-ending move (end turn AND attack) carries a
large fixed penalty that has nothing to do with the move's quality. Measured on v24,
120 games: median delta -3.093 when control passed vs -0.347 when it did not, i.e. a
**-2.746 pp** systematic offset. Uncorrected, a single margin either throws away half
of all attacks (margin 3.09 -> attack 48.7% adopted) or stops filtering at all
(margin 6.19 -> retreat 96.2%).

This is the same job the handcrafted scorer's `--eval-margin 1.0` did (cancel the
opp-hand+1 artifact), but applied CONDITIONALLY rather than globally, which is what the
handcrafted version could not distinguish."""


def _state(obs):
    try:
        st = to_observation_class(obs).current
        return st if (st and len(st.players or []) == 2) else None
    except Exception:
        return None


def _eval(obs, me):
    """State score from player ``me``'s view, or None.

    Handcrafted: eval_state scale (1 prize = 1000). Learned: win probability in
    PERCENTAGE POINTS. The two scales are unrelated -- --eval-margin/--eval-temp are
    calibrated per scorer, never shared."""
    st = _state(obs)
    if st is None:
        return None
    try:
        return _SCORER(st, me) if _SCORER is not None else evaluate(st, me)
    except Exception:
        return None

LT_ATTACK, LT_RESULT = 15, 23   # LogType markers that end a reasoning window
SEP = " ACT "                   # separates reasoning from action in the "reason" target


def _read_game(path):
    """Yield (header, [steps]) for each game block in a .jsonl.gz file."""
    header, steps = None, []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("kind") == "game":
                if header is not None:
                    yield header, steps
                header, steps = rec, []
            else:
                steps.append(rec)
    if header is not None:
        yield header, steps


def _action_target(step):
    obs = step["obs"]
    return ",".join(encode_option(o, obs) for o in step.get("chosen", []))


def _executed_chosen(step):
    """The option dicts of the move that was actually EXECUTED (== heuristic choice
    unless this step explored, in which case the plausible-random move)."""
    obs = step["obs"]
    opts = (obs.get("select") or {}).get("option") or []
    idx = step.get("executed") if step.get("explored") else step.get("action", [])
    return [opts[j] for j in (idx or []) if 0 <= j < len(opts)]


def _executed_target(step):
    obs = step["obs"]
    return ",".join(encode_option(o, obs) for o in _executed_chosen(step))


def _executed_indices(step):
    """Original option indices the agent picked, in recorded order."""
    return (step.get("executed") if step.get("explored") else step.get("action")) or []


TARGET_MODE = "action"          # "action" = encode_option string, "index" = menu position


def _idx_target(step, remap=None):
    """The chosen option's position in the RENDERED menu, as a string.

    `remap` is the substate's remaining-original-index list for a multi-pick step, because the
    substate menu is a subset and renumbers. Returns None when the position cannot be resolved,
    so the caller can drop the record rather than emit a wrong label."""
    idx = _executed_indices(step)
    if len(idx) != 1:
        return None
    j = idx[0]
    if remap is not None:
        try:
            j = list(remap).index(j)
        except ValueError:
            return None
    return str(j)



def _dname(header, p):
    """Deck NAME for player p. gen_selfplay writes header["agents"] = {"0": nameA, "1": nameB}."""
    a = (header or {}).get("agents") or {}
    return a.get(str(p)) or a.get(p)

def _multipick_pairs(step, deck_ids, deck_name=None):
    """Decompose a multi-pick (maxCount>=2) into a SEQUENCE of single-pick samples:
    for each picked option (in recorded order) a (prompt, target) where the prompt's
    menu is the not-yet-picked options and the target is the chosen option's encoding;
    then, if the agent picked fewer than max (an 'up to k' select), a final STOP sample.
    Same substate builder as inference, so train and inference prompts match."""
    obs = step["obs"]
    sel = obs.get("select") or {}
    opts = sel.get("option") or []
    lo, hi = sel.get("minCount", 1) or 0, sel.get("maxCount", 1) or 1
    order = [i for i in _executed_indices(step) if 0 <= i < len(opts)]
    pairs, picked = [], []
    for pos_i in order:
        sub, _remaining, _stop = multipick_substate(obs, picked)
        _t = (str(_remaining.index(pos_i)) if TARGET_MODE == "index"
              else encode_option(opts[pos_i], obs))
        pairs.append((_ser_cur(sub, deck_ids, deck_name), _t))
        picked.append(pos_i)
    if lo <= len(picked) < hi:                 # heuristic chose to stop early
        sub, _remaining, allow_stop = multipick_substate(obs, picked)
        if allow_stop:
            pairs.append((_ser_cur(sub, deck_ids, deck_name), STOP))
    return pairs


def _s1_step(step, by_i, me):
    """The state that captures this move's FULL effect: skip the move's own
    sub-selections (player==me, not MAIN) so a mid-combo state (card spent, payoff
    not yet applied) is never used, then take the next decision -- the player's
    next MAIN this turn, or the first state after control leaves them if the move
    ended the turn. None if the game ended on this move (terminal)."""
    j = step["i"] + 1
    while j in by_i and by_i[j].get("player") == me and not by_i[j].get("is_main"):
        j += 1
    return by_i.get(j)


def _prefetch_scores(steps, by_i, adopt):
    """Score every (S0, S1) this GAME's adoption filter will need in ONE predict call.

    The pairs depend only on the log, never on a decision we have not made yet, so they
    can all be computed up front. Per candidate, `_SCORER.pair()` cost 6.64 ms -- 146x
    the handcrafted evaluate() it replaced -- and that single factor is what pushed the
    Kaggle build past the 12 h cap from v32 onward. Batched per game (~150 states) the
    predict drops to ~0.05 ms/state. Returns {(step_i, player): (e0, e1)} or None when
    the learned scorer isn't in use."""
    if _SCORER is None or adopt != "eval":
        return None
    items = []
    for s in steps:
        p = s.get("player")
        if p not in (0, 1) or not s.get("is_main"):
            continue
        s1 = _s1_step(s, by_i, p)
        if s1 is None:                   # terminal -- caller adopts without a score
            continue
        items.append(((s["i"], p), p, _state(s["obs"]), _state(s1["obs"])))
    return _SCORER.batch_pairs(items)


def _eval_delta(step, by_i, me, scores=None):
    """Evaluator change from this state to the fully-resolved post-move state, from
    the deciding player's view. Returns (delta, s1); delta is None if terminal or
    either state can't be scored."""
    s1 = _s1_step(step, by_i, me)
    if s1 is None:
        return None, None                # terminal (winning move) -- handled by caller
    if scores is not None:               # batched up front by _prefetch_scores
        e0, e1 = scores.get((step["i"], me), (None, None))
    elif _SCORER is not None:            # one predict call for both states
        e0, e1 = _SCORER.pair(_state(step["obs"]), _state(s1["obs"]), me)
    else:
        e0, e1 = _eval(step["obs"], me), _eval(s1["obs"], me)
    if e0 is None or e1 is None:
        return None, s1
    d = e1 - e0
    if _TURN_BOUNDARY and s1.get("player") is not None and s1.get("player") != me:
        d += _TURN_BOUNDARY
    return d, s1


def _reasoning(psteps, k, outcome, window):
    """Compact event-anchored future window: the SAME player's future-step logs up
    to the next ATTACK/RESULT (inclusive), capped at `window` steps, then the
    outcome line (win/loss from that player's view)."""
    parts = []
    for s in psteps[k + 1: k + 1 + window]:
        logs = s["obs"].get("logs") or []
        r = render_logs(logs)
        if r:
            parts.append(r)
        if any(lg.get("type") in (LT_ATTACK, LT_RESULT) for lg in logs):
            break
    body = " ".join(parts)
    return (body + " " if body else "") + outcome


def _nopt(s):
    return s.get("n_options") or len((s["obs"].get("select") or {}).get("option") or [])


_ACC_KEYS = ("cand", "adopt", "rej_neg", "no_eval", "explored_cand", "explored_adopt",
             "heur_cand", "heur_adopt", "terminal_adopt", "win_cand", "win_adopt",
             "lose_cand", "lose_adopt", "sub_adopt", "soft_admit")
_DELTA_RESERVOIR = 5000     # per shard; enough for stable quantiles once merged


def _shard_paths(out_dir, tag, idx):
    p = os.path.join(out_dir, f"{tag}.shard-{idx:04d}.jsonl.gz")
    return p, p + ".stats.json"


def _build_shard(job):
    """Process ONE matchup log into its own shard + stats sidecar.

    Sharding is what makes the build both parallel and crash-safe: a shard lands the
    moment its matchup finishes, so a killed run (Kaggle's 12 h CPU cap) keeps every
    completed shard and the next run resumes on the rest.  The old single-writer build
    only moved its output after returning, so a kill lost the entire run.
    """
    idx, path, out_dir, tag, window, modes, adopt, margin, temp, seed = job
    shard, statp = _shard_paths(out_dir, tag, idx)
    if os.path.exists(shard) and os.path.exists(statp):     # resume: already done
        with open(statp) as f:
            st = json.load(f)
        st["resumed"] = True
        return st

    st = {k: 0 for k in _ACC_KEYS}
    st.update(n_games=0, n_main=0, n_samples=0, skipped=0, resumed=False,
              delta_sum=0.0, delta_n=0, deltas=[], plen_sum=0, plen_n=0,
              plen_max=0, plen_min=1 << 30, preview=[], path=os.path.basename(path))
    # seed PER SHARD so the soft threshold stays reproducible independent of worker order
    rng = random.Random(seed + idx)
    tmp = shard + ".part"
    with gzip.open(tmp, "wt", encoding="utf-8") as out:
            for header, steps in _read_game(path):
                winner = header.get("winner")
                if winner is None:                 # draw / timeout: no outcome to key on
                    st["skipped"] += 1
                    continue
                st["n_games"] += 1
                by_i = {s["i"]: s for s in steps}   # for the post-move state S1 (MAIN delta)
                # each player's FULL deck (visible only at turn 0) -> STABLE glossary prefix (v2);
                # take the longest deck seen per side (= the initial 60 before any draw).
                # Scanning the per-step obs for this is a DEAD END: an observation carries only
                # `deckCount`, never the list, so the scan below always produced [] and every
                # prompt rendered WITHOUT the DECK[...] segment -- verified at 0.0% of 20,000
                # records. It is the same hole that made every build_rerank record deck-less
                # while inference passed the real 60 cards. gen_selfplay writes the real lists
                # into the game header (`"decks": {"0": d0, "1": d1}`), so read them from there.
                _hd = (header or {}).get("decks") or {}
                game_decks = {0: list(_hd.get("0") or _hd.get(0) or []),
                              1: list(_hd.get("1") or _hd.get(1) or [])}
                if not (game_decks[0] and game_decks[1]):      # pre-schema-1 logs: fall back
                    for _s in steps:
                        for _i, _pl in enumerate(((( _s.get("obs") or {}).get("current") or {}).get("players") or [])[:2]):
                            _d = [c["id"] for c in (_pl.get("deck") or [])]
                            if len(_d) > len(game_decks.get(_i, [])):
                                game_decks[_i] = _d
                scores = _prefetch_scores(steps, by_i, adopt)   # ONE predict per game
                psteps = {0: [], 1: []}
                pos = {}
                for s in steps:
                    p = s.get("player")
                    if p in (0, 1):
                        pos[s["i"]] = len(psteps[p])
                        psteps[p].append(s)
                outcome = {}
                for p in (0, 1):
                    taken = 6 - int(header["prize_remaining"].get(str(p), 6))
                    outcome[p] = (f"=> win +{taken}pz" if p == winner
                                  else f"=> loss +{taken}pz")

                for s in steps:
                    p = s.get("player")
                    if p not in (0, 1):
                        continue
                    is_main = s.get("is_main")
                    emit_reason = is_main
                    if is_main:
                        if adopt == "winner" and p != winner:
                            continue
                        st["n_main"] += 1
                        if adopt == "eval":
                            st["cand"] += 1
                            won = (p == winner)
                            st["win_cand" if won else "lose_cand"] += 1
                            explored = bool(s.get("explored"))
                            st["explored_cand" if explored else "heur_cand"] += 1
                            delta, s1 = _eval_delta(s, by_i, p, scores)
                            if delta is None:
                                keep = True
                                st["terminal_adopt" if s1 is None else "no_eval"] += 1
                            else:
                                st["delta_sum"] += delta; st["delta_n"] += 1
                                if len(st["deltas"]) < _DELTA_RESERVOIR:
                                    st["deltas"].append(delta)
                                if delta >= -margin:
                                    keep = True
                                elif temp > 0:
                                    # SOFT threshold: admit borderline-negative moves
                                    # with probability exp(-slack/temp) (near threshold
                                    # ~always, far-negative blunders ~never) for diversity
                                    keep = rng.random() < math.exp((delta + margin) / temp)
                                    if keep:
                                        st["soft_admit"] += 1
                                else:
                                    keep = False
                                if not keep:
                                    st["rej_neg"] += 1
                            if not keep:
                                continue
                            st["adopt"] += 1
                            st["win_adopt" if won else "lose_adopt"] += 1
                            st["explored_adopt" if explored else "heur_adopt"] += 1
                            action = _executed_target(s)
                        else:                                  # legacy winner-only
                            action = _action_target(s)
                        pairs = [(_ser_cur(s["obs"], game_decks.get(p), _dname(header, p)),
                                  (_idx_target(s) if TARGET_MODE == "index" else action))]
                    else:                                      # sub-selection
                        if adopt == "winner" or _nopt(s) < 2:  # forced / legacy: skip
                            continue
                        sel = s["obs"].get("select") or {}
                        lo, hi = sel.get("minCount", 1) or 0, sel.get("maxCount", 1) or 1
                        if lo == 1 and hi == 1:                # exactly one -> single sample
                            pairs = [(_ser_cur(s["obs"], game_decks.get(p), _dname(header, p)),
                                      (_idx_target(s) if TARGET_MODE == "index"
                                       else _executed_target(s)))]
                        else:                                  # multi / optional -> sequential
                            pairs = _multipick_pairs(s, game_decks.get(p), _dname(header, p))
                        st["sub_adopt"] += len(pairs)

                    k = pos[s["i"]]
                    pairs = [(a, b) for a, b in pairs if b is not None]
                    for prompt_body, act_target in pairs:
                        for mode in modes:
                            if mode == "reason" and not emit_reason:
                                continue
                            if mode == "act":
                                target = act_target
                            else:  # reason (MAIN only)
                                target = _reasoning(psteps[p], k, outcome[p], window) + SEP + act_target
                            prompt = f"[{mode.upper()}]\n{prompt_body}"
                            rec = {"game_id": header["game_id"], "i": s["i"],
                                   "mode": mode, "prompt": prompt, "target": target,
                                   "kind": "main" if is_main else "sub"}
                            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                            st["n_samples"] += 1
                            L = len(prompt)
                            st["plen_sum"] += L; st["plen_n"] += 1
                            st["plen_max"] = max(st["plen_max"], L)
                            st["plen_min"] = min(st["plen_min"], L)
                            if len(st["preview"]) < 3 and not is_main:
                                st["preview"].append(rec)

    os.replace(tmp, shard)              # atomic: a shard only exists once complete
    with open(statp, "w") as f:         # stats sidecar written AFTER -> resume is safe
        json.dump(st, f)
    return st


def _init_worker(value_model, turn_boundary, target_mode="action"):
    """Fit this worker's own scorer. Never inherit one across the process boundary.

    The previous version deliberately used `fork` so children would inherit the parent's
    already-fitted ValueScorer and skip a ~25-100 s refit each. That DEADLOCKED on
    Kaggle: sklearn/numpy start OpenMP threads during the parent's fit, fork() carries
    over only the calling thread, and the child then blocks forever on a mutex held by a
    thread that does not exist. Evidence: after 12 h the run had produced 4 `.part` files
    -- exactly one per worker, all on their first matchup, all empty -- and 0 of 1770
    shards. Saving 4 refits cost the entire run. Workers are spawned now, and single
    threading (below) also stops 4 workers x N BLAS threads oversubscribing 4 cores."""
    global TARGET_MODE          # workers are SPAWNED, so the parent's global never arrives
    TARGET_MODE = target_mode
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    global _SCORER, _TURN_BOUNDARY
    if value_model:
        from value_score import ValueScorer
        _SCORER = ValueScorer(value_model)
        _TURN_BOUNDARY = turn_boundary


def build(tag, out_dir, window, main_only, modes, adopt, margin, temp=0.0, seed=0,
          workers=0, merge=True, value_model="", turn_boundary=0.0):
    """The LM must make EVERY selection, so BOTH main moves and sub-selections (with
    a real choice, n_options >= 2) are training targets. Prompts are STATELESS
    (current board only, no episode history -- serialize_stateless).

    adopt: 'eval'  -> BOTH players. MAIN: label = the EXECUTED move (heuristic OR
                      plausible-random), adopted iff its evaluator delta is >= -margin
                      ('not negative'; removes winner-deck bias, admits good random
                      moves). SUB-SELECT: adopted as-is (heuristic label) -- there is
                      no exploration on sub-selects and the per-move evaluator delta is
                      unreliable mid-combo, so it is not gated.
           'winner' -> legacy: winner-only MAIN, heuristic label.

    Runs one worker per matchup file.  Output is SHARDED (`<tag>.shard-NNNN.jsonl.gz`,
    a valid gzip each) and merged at the end unless merge=False; leaving the shards is
    what lets a long build survive a kill and resume."""
    in_dir = os.path.join(ROOT, "data", "selfplay", tag)
    files = sorted(glob.glob(os.path.join(in_dir, "*__vs__*.jsonl.gz")))
    if not files:
        raise SystemExit(f"no log files in {in_dir}")
    os.makedirs(out_dir, exist_ok=True)
    # write GZIPPED directly: the stateless full-rules prompts are ~2.9 KB each, so a
    # full run's plain jsonl would be ~25 GB (overflows Kaggle's 20 GB disk); streaming
    # gzip keeps it ~0.75 GB and the uncompressed file never materializes.
    out_path = os.path.join(out_dir, f"{tag}.jsonl.gz")

    workers = workers or (os.cpu_count() or 1)
    jobs = [(i, p, out_dir, tag, window, modes, adopt, margin, temp, seed)
            for i, p in enumerate(files)]
    print(f"build: {len(files)} matchups x {workers} workers -> {out_dir}", flush=True)

    t0 = time.time()
    results = []
    if workers > 1:
        import multiprocessing as mp
        # Each worker refits from the npz (~30-90 s, once). Handing them a pickle of a
        # parent-fitted model was TRIED and REVERTED: the one measurement of it ran the
        # same 12-game smoke in 4.4 h instead of 141 s, with byte-identical output and no
        # explanation found (22 idle cores, load 6.3, no runaway process). An unexplained
        # 100x is not an optimization. Against 1770 matchups the refit is rounding error.
        # ALWAYS spawn -- see _init_worker. fork after the parent has fitted the scorer
        # (numpy/sklearn OpenMP threads) hangs every child on its first matchup.
        ctx = mp.get_context("spawn")
        with ctx.Pool(workers, initializer=_init_worker,
                      initargs=(value_model, turn_boundary, TARGET_MODE)) as pool:
            for st in pool.imap_unordered(_build_shard, jobs):
                results.append(st)
                _progress(results, len(files), t0)
    else:
        for job in jobs:
            results.append(_build_shard(job))
            _progress(results, len(files), t0)

    A = {k: sum(r[k] for r in results) for k in _ACC_KEYS}
    n_games = sum(r["n_games"] for r in results)
    n_main = sum(r["n_main"] for r in results)
    n_samples = sum(r["n_samples"] for r in results)
    skipped = sum(r["skipped"] for r in results)
    deltas = [d for r in results for d in r["deltas"]]
    delta_sum = sum(r["delta_sum"] for r in results)
    delta_n = sum(r["delta_n"] for r in results)
    plen_n = sum(r["plen_n"] for r in results)
    samples_preview = next((r["preview"] for r in results if r["preview"]), [])

    print(f"games={n_games}  skipped(draw/timeout)={skipped}")
    print(f"MAIN candidates={n_main}  sub-selects adopted={A['sub_adopt']}  ->  "
          f"samples={n_samples} (modes={','.join(modes)}, adopt={adopt})")
    if adopt == "eval":
        print(f"adoption: {A['adopt']}/{A['cand']} MAIN moves "
              f"({100*A['adopt']/max(1,A['cand']):.1f}%)  "
              f"rejected(negative)={A['rej_neg']}  soft-admit(temp={temp})={A['soft_admit']}  "
              f"terminal_adopt={A['terminal_adopt']}  unscorable_adopt={A['no_eval']}")
        print(f"  by side: WINNER {A['win_adopt']}/{A['win_cand']} "
              f"({100*A['win_adopt']/max(1,A['win_cand']):.1f}%)   "
              f"LOSER {A['lose_adopt']}/{A['lose_cand']} "
              f"({100*A['lose_adopt']/max(1,A['lose_cand']):.1f}%)  "
              f"<- both sides contribute (deck-bias removed)")
        print(f"  by source: heuristic {A['heur_adopt']}/{A['heur_cand']} "
              f"({100*A['heur_adopt']/max(1,A['heur_cand']):.1f}%)   "
              f"explored(random) {A['explored_adopt']}/{A['explored_cand']} "
              f"({100*A['explored_adopt']/max(1,A['explored_cand']):.1f}%)")
        if deltas:
            # mean is exact (running sum); quantiles come from the per-shard reservoirs
            d2 = sorted(deltas)
            q = lambda p: d2[min(len(d2) - 1, int(len(d2) * p))]
            print(f"  eval delta: mean {delta_sum/max(1,delta_n):+.1f}  "
                  f"p10 {q(.10):+.1f}  p50 {q(.50):+.1f}  p90 {q(.90):+.1f}  "
                  f"(margin={margin}, WIN={WIN:g}, n={delta_n}, quantiles from {len(d2)} sampled)")
    if plen_n:
        print(f"prompt chars: mean {sum(r['plen_sum'] for r in results)/plen_n:.0f}  "
              f"max {max(r['plen_max'] for r in results)}  "
              f"min {min(r['plen_min'] for r in results if r['plen_n'])}")

    if merge:
        # gzip members concatenate: `cat a.gz b.gz` is one valid stream, so the merge is
        # a byte copy -- no decompress/recompress pass over ~0.75 GB.
        shards = [_shard_paths(out_dir, tag, i)[0] for i in range(len(files))]
        with open(out_path, "wb") as w:
            for sh in shards:
                with open(sh, "rb") as r:
                    shutil.copyfileobj(r, w, 1 << 20)
        for i in range(len(files)):
            for p in _shard_paths(out_dir, tag, i):
                os.remove(p)
        print(f"-> {out_path}  ({os.path.getsize(out_path)/2**20:.0f} MB, {len(shards)} shards merged)")
    else:
        print(f"-> {out_dir}/{tag}.shard-*.jsonl.gz  ({len(files)} shards, NOT merged)")
    for rec in samples_preview:
        print(f"\n--- sample (kind={rec.get('kind')}, mode={rec['mode']}) ---")
        print("PROMPT:", rec["prompt"][:700])
        print("TARGET:", rec["target"][:200])


def _progress(results, total, t0):
    """Per-shard progress with an ETA.  The old build printed NOTHING until it finished,
    so a run heading for the 12 h wall was indistinguishable from a healthy one."""
    n = len(results)
    if n % 10 and n != total:
        return
    el = time.time() - t0
    done = sum(1 for r in results if not r.get("resumed"))
    print(f"[{n}/{total}] {sum(r['n_samples'] for r in results)} samples  "
          f"{el/60:.1f} min elapsed  ETA {el/max(1,n)*(total-n)/60:.1f} min"
          + (f"  ({n-done} resumed)" if n != done else ""), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True, help="gen_selfplay run tag under data/selfplay/")
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "sft"))
    ap.add_argument("--window", type=int, default=6, help="max future winner-steps in reasoning")
    ap.add_argument("--main-only", action="store_true", default=True,
                    help="emit samples only at MAIN selections (sub-selects -> heuristic)")
    ap.add_argument("--modes", default="act,reason", help="comma list: act, reason")
    ap.add_argument("--target-mode", choices=["action", "index"], default="action",
                    help="'action' emits encode_option(chosen); 'index' emits the chosen "
                         "option's position in the rendered menu (1-2 tokens, and one forward "
                         "then yields a distribution over ALL candidates -- what distillation "
                         "into the cross-encoder needs)")
    ap.add_argument("--value-model", default="",
                    help="path to value_data.npz / value_model.pkl / their dir; "
                         "enables the LEARNED scorer (scale = win-prob percentage points)")
    ap.add_argument("--turn-boundary", type=float, default=2.75,
                    help="pp added back when a move hands control over (learned scorer "
                         "only); measured -2.746, see calibrate_value_margin.py")
    ap.add_argument("--adopt", choices=["eval", "winner"], default="eval",
                    help="eval: label=EXECUTED move, adopt iff evaluator delta >= -margin "
                         "(heuristic AND plausible-random moves); winner: legacy heuristic label")
    ap.add_argument("--eval-margin", type=float, default=1.0,
                    help="'not negative' tolerance: adopt a move iff eval(S1)-eval(S0) >= -margin. "
                         "Default 1.0 cancels the turn-boundary artifact (a move that ends the turn "
                         "lands on the opponent's post-draw state, a fixed -1.0 from op hand +1); it "
                         "still rejects real tempo losers (retreat, card-negative plays)")
    ap.add_argument("--eval-temp", type=float, default=5.0,
                    help="SOFT-threshold temperature for diversity: a below-margin move is admitted "
                         "with prob exp((delta+margin)/temp) -- borderline moves ~always, blunders "
                         "~never. 0 = hard threshold. Default 5.0 (+~4%% moves, -33 blunder P~0.002)")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for the soft threshold (reproducible)")
    ap.add_argument("--workers", type=int, default=0,
                    help="parallel worker processes, one matchup each (0 = os.cpu_count()). "
                         "The build is pure CPU and embarrassingly parallel across matchups")
    ap.add_argument("--no-merge", dest="merge", action="store_false",
                    help="leave the per-matchup shards instead of concatenating them. Use on "
                         "Kaggle with --out pointed at the persisted output dir: every finished "
                         "shard survives a 12 h-limit kill and the next run resumes on the rest")
    args = ap.parse_args()
    global TARGET_MODE
    TARGET_MODE = args.target_mode
    if args.value_model:
        global _SCORER, _TURN_BOUNDARY
        _TURN_BOUNDARY = args.turn_boundary
        # Under spawn the workers fit their own; fitting here too would just burn a
        # refit in the parent, which then sits idle collecting results.
        if (args.workers or os.cpu_count() or 1) == 1:
            from value_score import ValueScorer
            _SCORER = ValueScorer(args.value_model)
            print(f"scorer: LEARNED value function via {_SCORER.source} "
                  f"(scale = win-prob pp; margin/temp must be calibrated in pp)")
        else:
            print(f"scorer: LEARNED value function, refit per worker from "
                  f"{os.path.basename(args.value_model)} (parent does not fit)")
    build(args.tag, args.out, args.window, args.main_only, args.modes.split(","),
          args.adopt, args.eval_margin, args.eval_temp, args.seed,
          workers=args.workers, merge=args.merge,
          value_model=args.value_model, turn_boundary=args.turn_boundary)


if __name__ == "__main__":
    main()
