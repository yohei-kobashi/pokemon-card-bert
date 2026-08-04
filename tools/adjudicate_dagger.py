#!/usr/bin/env python3
"""Step 1 of the counterfactual-DAgger proposal: is engine_v2's DAgger label actually right?

DAgger keeps the decisions where the LM disagrees with engine_v2 and asserts engine_v2 was
right. engine_v2 is a 62.7% heuristic, and on attach decisions it captures only 48% of the
available value ([[attach-value-measured]]), so some unknown share of those rows teach the
WRONG move. This measures the share, by adjudicating each disagreement with counterfactual
playouts instead of assuming.

It answers only "how often is the label wrong"; it changes no data. Pre-registered rule, set
before looking: share(LM's move better) < 15% with a small mean gap -> the label is healthy and
the counterfactual branch is dropped; > 25% -> label correction is worth building.

Method. At a disagreement we are standing on the live state, so no replay is needed: branch on
exactly two candidates -- engine_v2's pick and the LM's -- with tools/rl_branch.branch_values,
which determinizes the hidden cards and shares each determinization across both candidates
(common random numbers, so neither branch wins by drawing a better deck).

Two biases to keep in mind when reading the output, both inherent to playout adjudication:
  * playouts run engine_v2 on BOTH sides, so Q is "value if engine_v2 plays the rest". A move
    that only pays off with a better follow-up cannot score, which biases the verdict toward
    moves engine_v2 can convert -- i.e. conservative about calling the LM right.
  * rl_branch raises when the visible-card accounting does not reconcile (~3.6% of decisions);
    those are skipped, not scored.
"""
import argparse
import collections
import gzip
import json
import os
import random
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "cg-lib"), os.path.join(ROOT, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--decks", required=True, help="comma list")
    ap.add_argument("--model", required=True, help="hf:<dir> | qwen:<dir> | noisy:<q> | engine")
    ap.add_argument("--games", type=int, default=30, help="per deck; seats alternate")
    ap.add_argument("--playouts", type=int, default=8)
    ap.add_argument("--rate", type=float, default=1.0, help="fraction of disagreements to judge")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--engine-seed-base", type=int, default=0, help="0 = unseeded engine")
    ap.add_argument("--mirror-so", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import library
    import rl_branch
    from lm.action_token import dedup_options
    from lm.actions import encode_option
    from lm.agent import make_lm_agent
    from tools.mirror_match import make_agent

    decks = [d.strip() for d in args.decks.split(",") if d.strip()]
    tuning = json.load(open(os.path.join(ROOT, "agents", "tuning.json")))
    rng = random.Random(args.seed)
    st = collections.Counter()
    rows = []
    t0 = time.time()

    eng = None
    if args.engine_seed_base:
        from tools.mirror_env import DEFAULT_SO, MirrorEngine
        eng = MirrorEngine(args.mirror_so or DEFAULT_SO)
        battle_select, battle_finish = eng.select, eng.finish
    else:
        from cg.game import battle_finish, battle_select, battle_start

    with gzip.open(args.out, "wt") as out:
        for di, deck in enumerate(decks):
            ids = [int(x) for x in open(library.deck_path(deck)) if x.strip()]
            prof = tuning.get(deck, {})
            lm_agent, _sc = make_agent(args.model, deck, ids, prof)
            ref = make_lm_agent(ids, prof, model=None)
            opp = make_lm_agent(ids, prof, model=None)
            for g in range(args.games):
                lm_seat = g % 2
                if eng:
                    obs = eng.start(ids, ids, args.engine_seed_base + di * 1000 + g, mirror=0)
                else:
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
                        judged = False
                        if len(opts) >= 2 and pick_lm and pick_ref \
                                and pick_ref[0] < len(opts) and pick_lm[0] < len(opts):
                            raw = [encode_option(o, obs) for o in opts]
                            cands, pos, keys = dedup_options(raw, obs)
                            idx = {keys[p]: n for n, p in enumerate(pos)}
                            lab, mine = idx.get(keys[pick_ref[0]]), idx.get(keys[pick_lm[0]])
                            if len(cands) >= 2 and lab is not None and lab != mine:
                                st["disagree"] += 1
                                if rng.random() < args.rate:
                                    st["tried"] += 1
                                    judged = True
                                    try:
                                        q = rl_branch.branch_values(
                                            obs, ids, ids, lm_seat, [pick_ref, pick_lm],
                                            ref, opp, n_playouts=args.playouts, rng=rng)
                                    except Exception:
                                        q = None
                                    if q and q[0] is not None and q[1] is not None:
                                        st["resolved"] += 1
                                        pl = cur["players"][lm_seat]
                                        rec = {
                                            "deck": deck, "seat": lm_seat, "game": g,
                                            "q_ref": q[0], "q_lm": q[1],
                                            "prizes": len(pl.get("prize") or []),
                                            "n_options": len(cands),
                                            "cand_ref": str(cands[lab])[:60],
                                            "cand_lm": str(cands[mine])[:60] if mine is not None else "",
                                        }
                                        rows.append(rec)
                                        out.write(json.dumps(rec) + "\n")
                                    else:
                                        st["unresolved"] += 1
                        obs = battle_select(pick_lm)
                finally:
                    battle_finish()
            print("[%2d/%d] %-22s disagree %5d  judged %5d  resolved %5d  %.0fs"
                  % (di + 1, len(decks), deck, st["disagree"], st["tried"], st["resolved"],
                     time.time() - t0), flush=True)

    report(rows, st, time.time() - t0)


def _share(rows, key=lambda r: r["q_lm"] - r["q_ref"]):
    """-> (LM better, engine better, tie) as fractions of all rows."""
    d = [key(r) for r in rows]
    n = max(1, len(d))
    return (sum(1 for x in d if x > 0) / n,
            sum(1 for x in d if x < 0) / n,
            sum(1 for x in d if x == 0) / n)


def report(rows, st, secs):
    print("\n%s\n  disagreements %d | judged %d | resolved %d (%.1f%% of judged) | %.1f min"
          % ("=" * 78, st["disagree"], st["tried"], st["resolved"],
             100.0 * st["resolved"] / max(1, st["tried"]), secs / 60), flush=True)
    if not rows:
        print("  nothing resolved -- no verdict")
        return
    lm, en, tie = _share(rows)
    gaps = [r["q_lm"] - r["q_ref"] for r in rows]
    m = statistics.mean(gaps)
    se = statistics.stdev(gaps) / len(gaps) ** 0.5 if len(gaps) > 1 else 0.0
    print("\n  OVERALL (n=%d)" % len(rows))
    print("    LM's move better   %5.1f%%" % (100 * lm))
    print("    engine better      %5.1f%%" % (100 * en))
    print("    tie                %5.1f%%" % (100 * tie))
    print("    mean Q(LM) - Q(engine)  %+.4f +- %.4f   (t %+.2f)"
          % (m, se, m / se if se else 0.0))
    dec = [r for r in rows if r["q_lm"] != r["q_ref"]]
    if dec:
        l2, e2, _ = _share(dec)
        print("    among DECISIVE rows only (n=%d): LM %.1f%% / engine %.1f%%"
              % (len(dec), 100 * l2, 100 * e2))

    def slice_by(name, fn, buckets):
        print("\n  by %s" % name)
        for b in buckets:
            sel = [r for r in rows if fn(r) == b]
            if len(sel) < 20:
                continue
            l, e, t = _share(sel)
            g = statistics.mean([r["q_lm"] - r["q_ref"] for r in sel])
            print("    %-14s n=%5d   LM %5.1f%%  engine %5.1f%%  tie %5.1f%%   mean %+.3f"
                  % (b, len(sel), 100 * l, 100 * e, 100 * t, g))

    def pband(r):
        p = r["prizes"]
        return "2-4 prizes" if 2 <= p <= 4 else ("<=1 prize" if p <= 1 else ">=5 prizes")
    slice_by("prizes remaining", pband, ["<=1 prize", "2-4 prizes", ">=5 prizes"])

    def oband(r):
        n = r["n_options"]
        return "2-3 options" if n <= 3 else ("4-6 options" if n <= 6 else "7+ options")
    slice_by("option count", oband, ["2-3 options", "4-6 options", "7+ options"])

    def kind(r):
        c = (r["cand_ref"] + " " + r["cand_lm"]).lower()
        for k in ("attach", "retreat", "evolve", "end"):
            if k in c:
                return k
        return "other"
    slice_by("action kind (either side)", kind, ["attach", "retreat", "evolve", "end", "other"])

    print("\n  PRE-REGISTERED RULE: share(LM better) < 15%% -> label is healthy, drop the branch;"
          "\n                       > 25%% -> build label correction.   observed %.1f%%"
          % (100 * lm))


if __name__ == "__main__":
    main()
