#!/usr/bin/env python3
"""LM-vs-LM mirror games with EVERY decision logged and the outcome stamped on it.

WHY THIS EXISTS. The Phase-0 headroom audit prices a decision by branching it and letting
engine_v2 play the continuation, and that has two problems the user named: the value it
measures is Q^engine, not Q*, and -- worse -- `tools/diag_systematic.py` already documented
that the method is BLIND to a failure that repeats every turn. ns_zoroark goes 0-40 in the
mirror screen while no single decision measures as decisive (mean Q gap -0.0003). Fix one
instance, let engine_v2 rescue the rest, and the cost vanishes by construction.

So: look at whole games, with no engine_v2 anywhere in the measurement.

WHY MIRROR MAKES THIS A CONTROLLED EXPERIMENT AND NOT A CORRELATION HUNT. With --mirror both
seats get the same decklist AND the same shuffle order, and both seats are piloted by the SAME
model. Two seats of one game therefore differ in exactly two things: turn order, and the moves
they chose. Draw luck is not a confound -- it is literally identical for both players. That is
the whole reason to pay for mirror here.

Turn order IS a confound and it is handled by STRATIFYING, never by pooling: the analyser
compares a seat's own wins against its own losses. Pooling the seats would make
"what winners do" partly mean "what the player who moved first does".

WHAT IS NOT SOLVED. Splitting on the outcome is still conditioning on the future, so a
correlate can be an effect rather than a cause -- the winner attacks more BECAUSE it is
winning. `setup-execution-audit-and-budew-overattack` is the burned precedent: over-attack was
a symptom, and the generic promote-over-chip rule built on it helped feraligatr and regressed
dragapult. The analyser's answer is to report EARLY, PRIZE-MATCHED strata separately; this tool
just has to record enough for that, which is why every row carries turn and both prize counts.

    PYTHONPATH=cg-lib python3 tools/lm_mirror_log.py --model qwen:/root/out/i2_r7 \\
        --decks dragapult,marnie_grimmsnarl,... --games 40 --out /root/lmlog_r7.jsonl.gz
"""

import argparse
import gzip
import json
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "cg-lib"), os.path.join(ROOT, "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def kind_of(opt_text):
    """Coarse action kind from the rendered option ('atk:a123' -> 'atk'). Same rule as
    tools/diag_pilot.py, deliberately -- the two diagnostics have to bucket alike."""
    m = re.match(r"([a-z_]+)", opt_text or "")
    return m.group(1) if m else "?"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="hf:<dir> | qwen:<dir>")
    ap.add_argument("--decks", required=True, help="comma list")
    ap.add_argument("--protagonist", default="",
                    help="cross-deck mode: this deck plays EVERY deck in --decks instead of "
                         "itself. Seats alternate per game so the pair data is not a "
                         "first-player sample. Same-deck self-play only ever produced mirror "
                         "matchups, which do not occur on the ladder, and it left the prompt's "
                         "opponent-ID segment carrying no information (the opponent was always "
                         "our own list).")
    ap.add_argument("--games", type=int, default=40, help="per deck")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--mirror-so", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--trace-out", default="",
                    help="also write one line per GAME: the exact pick sequence (both seats) "
                         "plus per-decision seat/margin/runner-up. Enough to REPLAY the game "
                         "on the deterministic engine and branch any decision from the exact "
                         "state -- which is how the DPO pipeline moves states across machines "
                         "without shipping observation blobs. Draws are kept: the playout "
                         "measurement does not need the game's outcome.")
    ap.add_argument("--max-steps", type=int, default=4000)
    a = ap.parse_args()

    import library
    # The CANONICAL option renderer -- the same one tools/diag_pilot.py uses. The raw option
    # dict has no ready-made text field; reading o["text"] silently yields "" and every kind
    # buckets as "?", which is exactly what the first run of this tool did.
    from lm.actions import encode_option
    from mirror_env import DEFAULT_SO, MirrorEngine, engine_fingerprint, play
    from mirror_match import load_deck, make_agent

    so = a.mirror_so or DEFAULT_SO
    eng = MirrorEngine(so)
    tuning = json.load(open(os.path.join(ROOT, "agents", "tuning.json")))
    decks = [d for d in a.decks.split(",") if d]
    known = set(library.list_decks())
    missing = [d for d in decks if d not in known]
    if missing:
        sys.exit("unknown deck(s): %s" % ", ".join(missing))

    # Stamp the shuffle fingerprint. Two runs whose fingerprints differ did not play the same
    # games, so their logs must not be compared -- the same guard the screens carry.
    fp = engine_fingerprint(eng, load_deck(sorted(library.list_decks())[0]))
    print("[mirror] %s | shuffle fingerprint %s" % (so, fp), flush=True)

    out = gzip.open(a.out, "wt")
    trace_f = gzip.open(a.trace_out, "wt") if a.trace_out else None
    if trace_f:
        # The header carries the shuffle fingerprint so the replayer can refuse a mismatched
        # engine build instead of silently branching from different games.
        trace_f.write(json.dumps({"header": 1, "fp": fp, "model": a.model}) + "\n")
    n_rows = n_games = 0
    t0 = time.time()

    # Margins come from inside the scorer, not from re-scoring: wrap score() once and stash
    # each call's result. make_agent CACHES the scorer across decks, so the wrap must be
    # guarded or every deck would add another layer.
    score_calls = []

    def _tap(sc):
        if sc is None or getattr(sc, "_mirror_tap", False):
            return
        orig = sc.score

        def tapped(prompt, cands, obs):
            r = orig(prompt, cands, obs)
            score_calls.append(list(r) if r else [])
            return r
        sc.score = tapped
        sc._mirror_tap = True

    from lm.agent import _dedup

    if a.protagonist and a.protagonist not in known:
        sys.exit("unknown protagonist: %s" % a.protagonist)
    for deck in decks:
        ids = load_deck(deck)
        prof = tuning.get(deck, {})
        # ONE agent object, used for BOTH seats. Two separately-constructed agents would be the
        # same policy but not the same object, and any per-agent state (a scorer's cache, a
        # bank) would then differ between seats -- reintroducing an asymmetry the mirror exists
        # to remove.
        agent, _scorer = make_agent(a.model, deck, ids, prof)
        _tap(_scorer)
        # Cross-deck: the protagonist needs its OWN agent, because make_agent bakes the
        # decklist and the tuning profile in. The SCORER is cached across decks inside
        # make_agent, so both agents still share one policy object -- the property the
        # same-object rule above is really protecting.
        if a.protagonist:
            pids = load_deck(a.protagonist)
            pagent, _psc = make_agent(a.model, a.protagonist,
                                      pids, tuning.get(a.protagonist, {}))
            _tap(_psc)
        else:
            pids, pagent = ids, agent

        # Rows are buffered per game because `won` is only known at the end. A game is at most
        # a few hundred decisions, so this costs nothing and avoids a second pass over the file.
        pending = []
        tpicks, tmeta = [], []
        seat_wins = [0, 0]

        def _wrap(inner, dname=None):
            """Logging wrapper around ONE agent. Cross-deck needs two of these (one per
            decklist), so the closure that used to capture `agent` directly is now a factory."""

            def logging_agent(obs):
                return _log_step(inner, dname, obs)
            return logging_agent

        def _log_step(inner, dname, obs):
            cur = obs.get("current") or {}
            sel = obs.get("select") or {}
            raw = sel.get("option") or []
            opts = [encode_option(o, obs) for o in raw]
            del score_calls[:]
            pick = inner(obs)
            idx = pick[0] if isinstance(pick, (list, tuple)) and pick else pick
            chosen = opts[idx] if isinstance(idx, int) and 0 <= idx < len(opts) else ""
            yi = cur.get("yourIndex", 0)
            if trace_f is not None:
                # margin/alt only when this was a SINGLE-pick decision the scorer actually
                # decided (one score call, and the pick maps back to its argmax -- a pick that
                # does not is the engine fallback, whose margin would describe nothing).
                margin = alt_raw = None
                lo = sel.get("minCount", 1) or 0
                hi = sel.get("maxCount", 1) or 1
                if lo == 1 and hi == 1 and len(score_calls) == 1 \
                        and len(score_calls[0]) >= 2 and isinstance(idx, int):
                    uniq, pos = _dedup(opts, obs)
                    s = score_calls[0]
                    if len(s) == len(uniq):
                        order = sorted(range(len(s)), key=lambda i: -s[i])
                        if pos[order[0]] == idx:
                            margin = round(float(s[order[0]] - s[order[1]]), 4)
                            alt_raw = pos[order[1]]
                tpicks.append(list(pick) if isinstance(pick, (list, tuple)) else pick)
                tmeta.append([yi, margin, alt_raw, len(raw)])
            pl = cur.get("players") or [{}, {}]
            pz = [len((pl[i].get("prize") or [])) for i in (0, 1)] if len(pl) > 1 else [0, 0]
            pending.append({
                "deck": dname or deck, "seed": cur.get("_seed"), "seat": yi,
                "turn": cur.get("turn"),
                # BOTH prize counts. The analyser matches on the pair, not on ours alone: at
                # (2,5) we are winning and at (5,2) losing, and pooling those is how an
                # outcome-conditioned statistic turns into "winners are ahead".
                "my_pz": pz[yi], "op_pz": pz[1 - yi],
                "n_cand": len(opts),
                # the kinds OFFERED, so a take-rate can be conditioned on the menu. Without it
                # "the loser retreats less" cannot be separated from "the loser was offered
                # fewer retreats".
                "offered": sorted(set(kind_of(o) for o in opts)),
                "pick_kind": kind_of(chosen), "pick": chosen[:80],
            })
            return pick

        for g in range(a.games):
            s = a.seed + g
            del pending[:]
            del tpicks[:]
            del tmeta[:]
            # Alternate which seat the protagonist takes. Without this every cross-deck game
            # would put it first and the pairs would describe one seat's game only -- the
            # failure the seat-fair budget was added to fix, reintroduced upstream of it.
            if a.protagonist and g % 2:
                d0, d1, ids0, ids1 = deck, a.protagonist, ids, pids
                ag0, ag1 = agent, pagent
            else:
                d0, d1, ids0, ids1 = a.protagonist or deck, deck, pids, ids
                ag0, ag1 = pagent, agent
            r = play(eng, _wrap(ag0), _wrap(ag1), ids0, ids1, s, mirror=1,
                     max_steps=a.max_steps)
            n_games += 1
            if trace_f is not None and tpicks:
                trace_f.write(json.dumps({"deck0": d0, "deck1": d1, "seed": s, "result": r,
                                          "picks": tpicks, "meta": tmeta}) + "\n")
            if r not in (0, 1):
                continue                     # draw / timeout: no winner to split on
            seat_wins[r] += 1
            for row in pending:
                row["seed"] = s
                row["won"] = 1 if row["seat"] == r else 0
                out.write(json.dumps(row) + "\n")
                n_rows += 1
        print("  %-22s %d games | seat0 %d - seat1 %d wins | rows %d | %.0fs"
              % (deck, a.games, seat_wins[0], seat_wins[1], n_rows, time.time() - t0),
              flush=True)

    out.close()
    if trace_f:
        trace_f.close()
        print("traces -> %s" % a.trace_out)
    print("\nwrote %d rows from %d games to %s in %.1f min"
          % (n_rows, n_games, a.out, (time.time() - t0) / 60))
    if n_rows == 0:
        sys.exit("no rows: every game drew or timed out -- check the model spec and the engine")


if __name__ == "__main__":
    main()
