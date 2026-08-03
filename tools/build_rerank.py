"""Transform gen_selfplay logs -> CROSS-ENCODER RERANKER samples.

Emits, per decision, a LISTWISE record:
  {state, candidates: [action-string, ...], chosen: idx-into-candidates}
so a cross-encoder reranker (Alibaba-NLP/gte-reranker-modernbert-base) learns to score each
(state, candidate) pair and pick the winner's move. state == serialize_stateless(obs) (the
SAME the decoder used as its prompt); each candidate == encode_option(o, obs) (SAME action
encoding). SINGLE-pick and MULTI-pick decisions are BOTH emitted:

  - single (minCount==maxCount==1): candidates = all options, chosen = executed index.
  - multi / optional (choose k of n, or up-to-k): DECOMPOSE into a SEQUENCE of single-pick
    records EXACTLY as lm/agent does at inference -- at each step, sub-state =
    multipick_substate(obs, picked), candidates = the REMAINING options (+ a STOP candidate
    once min is satisfied), chosen = the winner's next pick (or STOP if they stopped early).
    Train and inference thus share the SAME decomposition, so no inference change is needed.

Usage:
    python tools/build_rerank.py --tag curengine_0724 --out /root/data/rerank
"""
import argparse
import glob
import gzip
import json
import os
import shutil
import sys
import time
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "cg-lib"), os.path.join(ROOT, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

from lm.serialize import serialize_stateless, multipick_substate, STOP  # noqa: E402
from lm.actions import encode_option                                    # noqa: E402


def _read_game(path):
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


def _executed_indices(step, label="heuristic"):
    """The option indices this record's LABEL points at.

    gen_selfplay eps-explores on MAIN selects: it EXECUTES a plausible alternative to visit
    diverse states but records ``action`` = the heuristic's own choice, saying so in its
    comment ("the recorded label stays the heuristic choice"). ``_plausible_legal`` builds
    that alternative by REMOVING the heuristic's choice, so on an explored step ``executed``
    is guaranteed to be a move engine_v2 rejected -- measured: 5.96% of all real-choice steps
    are explored and 100% of those differ.

    Training on ``executed`` therefore teaches the model to make the move the pilot we are
    imitating refused to make, and costs those rows on the top1 metric outright. ``label=
    "heuristic"`` keeps the diverse STATE that exploration bought and pairs it with the
    correct answer. ``label="executed"`` reproduces the old behaviour."""
    if label == "executed" and step.get("explored"):
        return step.get("executed") or []
    return step.get("action") or []


def _game_decks(header, steps):
    """{player_index: full 60-card id list}.

    READ IT FROM THE HEADER. The per-step obs has only ``deckCount`` -- there is no ``deck``
    field on a player -- so the old obs-scanning version always returned empty lists. That
    silently fed ``deck_ids=[]`` into serialize_stateless, which makes glossary_ids fall back
    to VISIBLE-ONLY (v1): the whole 'our deck first, for a stable KV-cache prefix' design
    never reached the training data, AND it diverged from inference, where make_lm_agent
    passes the real 60-card list. Train and deploy prompts must match."""
    decks = header.get("decks") or {}
    gd = {int(k): [int(x) for x in v] for k, v in decks.items() if v}
    if len(gd) == 2:
        return gd
    gd = {0: [], 1: []}                       # older logs: fall back to scanning the obs
    for _s in steps:
        players = (((_s.get("obs") or {}).get("current") or {}).get("players") or [])[:2]
        for _i, _pl in enumerate(players):
            _d = [c["id"] for c in (_pl.get("deck") or [])]
            if len(_d) > len(gd.get(_i, [])):
                gd[_i] = _d
    return gd


def _deck_names(header, path, decks=None):
    """{player_index: deck_name}, resolved by MATCHING THE 60-CARD LIST -- not by position.

    Both the game_id and the filename are ``<deck0>__vs__<deck1>``, but gen_selfplay
    ALTERNATES SEATS every game (measured: a perfect ``.X.X.X.X`` pattern in every matchup
    file), so ``{0: deck0, 1: deck1}`` was wrong in exactly 50% of games -- ``ID ME d_X``
    named the OPPONENT's deck half the time. That is not merely noise: at inference the
    submission hardcodes its own deck name, so the model was trained on a token whose
    meaning flips and deployed on one that never does.

    ``header["agents"]`` is the authoritative seat->name map and is preferred: it stays right
    even for OLD logs whose decklist has since been edited (v34_full, 2026-07-21, has ~3% of
    seats whose 60 cards no longer match today's decks/ file -- there, matching by list finds
    nothing and would silently fall back to the broken positional order).

    ``decks`` is _game_decks()'s {player_index: 60 ids}; without ``agents`` the name whose
    list matches wins. Falls back to positional order only when neither is usable (mirrors,
    where both names are the same anyway)."""
    agents = header.get("agents") or {}
    if len(agents) == 2:
        try:
            return {int(k): v for k, v in agents.items()}
        except (TypeError, ValueError):
            pass
    src = (header.get("game_id") or "").split("#")[0]
    if "__vs__" not in src:
        src = os.path.basename(path).split(".")[0]
    parts = src.split("__vs__")
    if len(parts) != 2:
        return {}
    pos = {i: n for i, n in enumerate(parts)}
    if not decks or len(decks) != 2 or parts[0] == parts[1]:
        return pos
    try:
        import library
        want = {n: Counter(library.read_deck(n)) for n in set(parts)}
    except Exception:
        return pos
    out = {}
    for p, ids in decks.items():
        got = Counter(int(x) for x in ids)
        hits = [n for n in parts if want.get(n) == got]
        if len(hits) != 1:
            return pos                        # unrecognisable list -> keep the old behaviour
        out[int(p)] = hits[0]
    return out if len(out) == 2 and out[0] != out[1] else pos


def _dedup_equivalent(raw, obs=None, state=None):
    """Candidate texts with acts-that-are-the-same collapsed to their first occurrence.

    Pass an `obs` (or a rendered `state`) to also collapse board slots the prompt shows
    identically -- three copies of one Basic on the bench are one attach target, not three.
    Without either this is the string-only collapse it always was.
    """
    from lm.action_token import dedup_options
    return dedup_options(raw, obs, state)[0]


def _canonical_index(cands, text, obs=None, state=None):
    """Where the chosen option ended up after the collapse. -> index or None

    Keyed the SAME way as the collapse: comparing by `equivalent` here while the collapse used
    the board would map the label to a candidate the dedup had already merged away.
    """
    from lm.action_token import canon_key, slot_map_from_obs, slot_map_from_state
    slots = (slot_map_from_obs(obs) if obs else
             (slot_map_from_state(state) if state else {}))
    want = canon_key(text, slots)
    for i, c in enumerate(cands):
        if c == text or canon_key(c, slots) == want:
            return i
    return None


def _emit(out, st, gid, i, state, raw, chosen_idx, kind, deck=None, opp=None,
          explored=False, obs=None):
    """Dedup candidates by the ACT they perform, remap chosen, drop if <2 remain, then write one
    listwise record.

    Deduping by exact text was not enough. `card:c305@DECK1` and `card:c305@DECK6` are two copies
    of one card in a shuffled pile, and `facedown:PRIZE2` and `facedown:PRIZE3` are two face-down
    prizes -- the same act, written differently. Keeping both put one in the positive slot and the
    other in the negatives, so 17.17% of records (measured on v39_0731: card 48,811, facedown
    18,657, energy 1,220) asked the model to rank apart two inputs that differ only in a position
    number. There is no feature that separates them, so the only way to drive that loss down is to
    memorise the number -- training pressure toward a spurious cue.

    It is also free speed at deployment, where the cross-encoder pays one forward pass per
    candidate: 5.84 -> 5.20 candidates per decision, and 5.65% of decisions collapse to a single
    candidate, i.e. there was never a choice to score.

    ``deck``/``opp`` name the PILOT and its opponent so the trainer can sample a balanced
    mix. Records are winner-only, so their natural distribution is proportional to WINS:
    620-38,987 records per deck, and a uniform sample leaves the weakest decks with almost
    no representation."""
    from lm.action_token import dedup_options
    cands, pos, keys = dedup_options(raw, obs)
    # Map the label through the SAME keys the collapse used. Re-deriving it with a second
    # comparison is how a label ends up pointing at a candidate that was merged away.
    want = keys[chosen_idx]
    chosen_new = next((n for n, p in enumerate(pos) if keys[p] == want), None)
    if len(cands) < 2 or chosen_new is None:
        st["n_nochoice"] += (len(cands) < 2)
        return
    out.write(json.dumps({"game_id": gid, "i": i, "state": state,
                          "candidates": cands, "chosen": chosen_new,
                          "kind": kind, "deck": deck, "opp": opp,
                          "explored": explored}, ensure_ascii=False) + "\n")
    st["n_records"] += 1
    st["n_cands"] += len(cands)


def _shard_paths(out_dir, tag, idx):
    p = os.path.join(out_dir, f"{tag}.rr-shard-{idx:04d}.jsonl.gz")
    return p, p + ".done"


def _stale(decks, names):
    """True if a seat's logged 60 cards no longer match that deck's current decks/ file.

    Older self-play (v34_full, 2026-07-21) predates several decklist rebuilds
    (mega_zygarde, mega_lucario). Those games teach a deck composition that no longer
    exists -- and the state literally renders it as ``DECK[...]`` -- so they are dropped.
    The whole GAME goes, not just that seat: if the OPPONENT's list is stale, our pilot
    played against a deck that is gone and the ID segment mis-identifies it too."""
    import library
    for p, ids in decks.items():
        nm = names.get(p)
        if not nm:
            return True
        try:
            if Counter(int(x) for x in ids) != Counter(library.read_deck(nm)):
                return True
        except Exception:
            return True
    return False


def _build_shard(job):
    idx, path, out_dir, tag, glossary, skip_stale, deck_mode, label, sides, dshuf = job
    # the prompt format is part of the model -- lm/agent must be given the SAME glossary
    # mode and deck_name at inference or train and deploy prompts diverge silently
    _ser = lambda o, p: serialize_stateless(  # noqa: E731
        o, deck_ids=gd.get(p), glossary=glossary, deck_name=dn.get(p),
        deck_mode=deck_mode, deck_shuffle=dshuf)
    shard, done = _shard_paths(out_dir, tag, idx)
    if os.path.exists(shard) and os.path.exists(done):
        with open(done) as f:
            return json.load(f)
    st = {"n_games": 0, "n_records": 0, "n_cands": 0, "n_single": 0, "n_mp": 0,
          "n_stop": 0, "mp_err": 0, "n_stale": 0, "n_nochoice": 0}
    tmp = shard + ".part"
    with gzip.open(tmp, "wt", encoding="utf-8") as out:
        for header, steps in _read_game(path):
            winner = header.get("winner")
            if winner is None:
                continue
            gd = _game_decks(header, steps)
            dn = _deck_names(header, path, gd)
            # a PERTURBED game logs a deliberately different 60 under the same deck name
            # (gen_selfplay --perturb); that is the point of the run, not an outdated list
            perturbed = any(int(v or 0) for v in (header.get("perturb") or {}).values())
            if skip_stale and not perturbed and len(gd) == 2 and _stale(gd, dn):
                st["n_stale"] += 1
                continue
            st["n_games"] += 1
            for s in steps:
                p = s.get("player")
                # WINNER-only halves the data to bias toward winning play. That trade is
                # right when the model is trying to beat its teacher and wrong when it is
                # 11pt BELOW it: the loser's moves are the SAME engine_v2 deciding, just in
                # losing positions -- positions the pilot also has to play well from.
                if p not in (0, 1) or (sides == "winner" and p != winner):
                    continue
                obs = s.get("obs") or {}
                sel = obs.get("select") or {}
                opts = sel.get("option") or []
                if len(opts) < 2:                            # no real choice
                    continue
                lo = sel.get("minCount", 1) or 0
                hi = sel.get("maxCount", 1) or 1
                order = [i for i in _executed_indices(s, label) if 0 <= i < len(opts)]
                expl = bool(s.get("explored"))
                kind = "main" if s.get("is_main") else "sub"
                gid, si = header["game_id"], s["i"]
                mine, theirs = dn.get(p), dn.get(1 - p)

                if hi == 1 and lo == 1:                       # SINGLE pick
                    if len(order) != 1:
                        continue
                    raw = [encode_option(o, obs) for o in opts]
                    _emit(out, st, gid, si, _ser(obs, p), raw, order[0], kind, mine,
                          theirs, expl, obs=obs)
                    st["n_single"] += 1
                    continue

                # MULTI / OPTIONAL -> decompose like lm/agent inference
                try:
                    picked, good = [], True
                    for pos_i in order:
                        sub, remaining, allow_stop = multipick_substate(obs, picked)
                        if pos_i not in remaining:
                            good = False; break
                        raw = [encode_option(opts[i], obs) for i in remaining]
                        chosen_local = remaining.index(pos_i)
                        if allow_stop:
                            raw = raw + [STOP]
                        _emit(out, st, gid, si, _ser(sub, p), raw, chosen_local, kind,
                              mine, theirs, expl, obs=sub)
                        st["n_mp"] += 1
                        picked.append(pos_i)
                    if good and lo <= len(picked) < hi:      # winner stopped early (incl. declined)
                        sub, remaining, allow_stop = multipick_substate(obs, picked)
                        if allow_stop and remaining:
                            raw = [encode_option(opts[i], obs) for i in remaining] + [STOP]
                            _emit(out, st, gid, si, _ser(sub, p), raw, len(raw) - 1, kind,
                                  mine, theirs, obs=sub)
                            st["n_stop"] += 1
                except Exception:
                    st["mp_err"] += 1
    os.replace(tmp, shard)
    with open(done, "w") as f:
        json.dump(st, f)
    return st


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True, help="gen_selfplay run tag under data/selfplay/")
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "rerank"))
    ap.add_argument("--suffix", default="", help="append to output filename (e.g. _mp)")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--no-merge", dest="merge", action="store_false")
    ap.add_argument("--glossary", default="full", choices=("full", "structured", "none"),
                    help="how much card-rules text to put in the state; see "
                         "lm.serialize.GLOSSARY_MODES. 'none' is 5x shorter (the deploy "
                         "cost knob for the cross-encoder), 'full' is the legacy format")
    ap.add_argument("--deck-shuffle", action="store_true",
                    help="permute DECK[]'s card order per decision. The canonical order is "
                         "the decklist FILE's order -- a perfect fingerprint that lets the "
                         "model recognise the deck without reading any card. See "
                         "lm.serialize.render_my_deck.")
    ap.add_argument("--sides", default="winner", choices=("winner", "both"),
                    help="whose decisions become records. 'both' doubles the data and drops "
                         "the outcome filter -- for IMITATION fidelity every engine_v2 "
                         "decision is a valid label regardless of who won the game.")
    ap.add_argument("--label", default="heuristic", choices=("heuristic", "executed"),
                    help="whose move is the answer on an eps-EXPLORED step. 'heuristic' = "
                         "engine_v2's own choice (correct for imitation); 'executed' = the "
                         "random alternative that was played (the old, wrong behaviour). "
                         "See _executed_indices.")
    ap.add_argument("--deck-mode", default="static", choices=("static", "remaining"),
                    help="DECK[]: the original 60 ('static') or what is still in the library "
                         "('remaining'). 'static' is 70%% redundant with the rest of the "
                         "state (85%% from turn 11) so a model can learn to ignore it; "
                         "'remaining' is available nowhere else. See render_my_deck.")
    ap.add_argument("--max-files", type=int, default=0, help="stop after N matchup files "
                    "(for a quick format probe instead of the full 1.5M-record build)")
    ap.add_argument("--keep-stale", dest="skip_stale", action="store_false",
                    help="keep games whose logged decklist no longer matches decks/")
    args = ap.parse_args()

    in_dir = os.path.join(ROOT, "data", "selfplay", args.tag)
    files = sorted(glob.glob(os.path.join(in_dir, "*__vs__*.jsonl.gz")))
    if not files:
        raise SystemExit(f"no log files in {in_dir}")
    if args.max_files:
        files = files[:args.max_files]
    os.makedirs(args.out, exist_ok=True)
    workers = args.workers or (os.cpu_count() or 1)
    shard_tag = args.tag + args.suffix
    jobs = [(i, p, args.out, shard_tag, args.glossary, args.skip_stale, args.deck_mode, args.label, args.sides, args.deck_shuffle)
            for i, p in enumerate(files)]
    print(f"build_rerank: {len(files)} matchups x {workers} workers -> {args.out}", flush=True)

    t0 = time.time()
    results = []
    if workers > 1:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        with ctx.Pool(workers) as pool:
            for st in pool.imap_unordered(_build_shard, jobs):
                results.append(st)
                if len(results) % 50 == 0 or len(results) == len(files):
                    el = time.time() - t0
                    nr = sum(r["n_records"] for r in results)
                    print(f"[{len(results)}/{len(files)}] {nr} records  {el/60:.1f} min  "
                          f"ETA {el/max(1,len(results))*(len(files)-len(results))/60:.1f} min", flush=True)
    else:
        for job in jobs:
            results.append(_build_shard(job))

    A = lambda k: sum(r.get(k, 0) for r in results)
    print(f"games={A('n_games')}  records={A('n_records')}  "
          f"(single={A('n_single')}, multipick-steps={A('n_mp')}, stop={A('n_stop')}, "
          f"mp_err={A('mp_err')})  avg cands/rec={A('n_cands')/max(1,A('n_records')):.1f}"
          f"  dropped-stale-decklist games={A('n_stale')}")

    if args.merge:
        out_path = os.path.join(args.out, f"{shard_tag}.rerank.jsonl.gz")
        shards = [_shard_paths(args.out, shard_tag, i)[0] for i in range(len(files))]
        with open(out_path, "wb") as w:
            for sh in shards:
                with open(sh, "rb") as r:
                    shutil.copyfileobj(r, w, 1 << 20)
        for i in range(len(files)):
            for p in _shard_paths(args.out, shard_tag, i):
                if os.path.exists(p):
                    os.remove(p)
        print(f"-> {out_path}  ({os.path.getsize(out_path)/2**20:.0f} MB)")


if __name__ == "__main__":
    main()
