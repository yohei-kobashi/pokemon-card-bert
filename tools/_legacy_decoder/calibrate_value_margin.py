"""Recalibrate build_sft's adoption threshold for the LEARNED value scorer.

The handcrafted evaluator's `--eval-margin 1.0` was principled ON ITS SCALE: a move that
ends the turn lands S1 on the opponent's post-draw state, so "opp hand +1" cost exactly
-1.0 (eval_state's `hand` weight), and margin=1.0 cancelled precisely that artifact.

The learned scorer is win probability in PERCENTAGE POINTS, so 1.0 means something
completely different and the old margin is meaningless. This measures the same artifact
on the new scale: the systematic offset a turn-ending move carries, per move type.

Recipe: margin ~= -(median delta of `end turn`), then check that attach/evolve/attack
land near 100% adoption while genuinely bad moves (retreat, card-negative plays) still
get filtered -- the same acceptance profile the handcrafted calibration produced.

Usage:
    python tools/calibrate_value_margin.py --value out/value --games 400
"""
import argparse, collections, gzip, json, os, random, statistics, sys, tarfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "cg-lib"), os.path.join(ROOT, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

import build_sft                                        # noqa: E402
from build_sft import _s1_step, _executed_chosen        # noqa: E402

OPT = {7: "play", 8: "attach", 9: "evolve", 10: "ability",
       12: "retreat", 13: "attack", 14: "end turn"}


def move_type(step):
    ch = _executed_chosen(step)
    if not ch:
        return "?"
    t = ch[0].get("type") if isinstance(ch[0], dict) else getattr(ch[0], "type", None)
    return OPT.get(int(t) if t is not None else -1, f"opt{t}")


def games_from(tar_path, n_games, seed=0):
    tf = tarfile.open(tar_path)
    names = [m.name for m in tf.getmembers() if m.name.endswith(".jsonl.gz")]
    random.Random(seed).shuffle(names)
    got = 0
    for nm in names:
        header, steps = None, []
        for line in gzip.open(tf.extractfile(nm), "rt"):
            rec = json.loads(line)
            if rec.get("kind") == "game":
                if header is not None and steps:
                    yield header, steps
                    got += 1
                    if got >= n_games:
                        return
                header, steps = rec, []
            elif rec.get("kind") == "step":
                steps.append(rec)
        if header is not None and steps:
            yield header, steps
            got += 1
            if got >= n_games:
                return


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tar", default="data/kaggle_out/v24_full/selfplay_v24_full_raw.tar")
    ap.add_argument("--value", default="", help="value artifact dir/npz; empty = handcrafted")
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--turn-boundary", type=float, default=0.0)
    args = ap.parse_args()

    if args.value:
        from value_score import ValueScorer
        build_sft._SCORER = ValueScorer(args.value)
        build_sft._TURN_BOUNDARY = args.turn_boundary
        print(f"scorer: LEARNED ({build_sft._SCORER.source}), scale = win-prob pp, "
              f"turn-boundary correction {args.turn_boundary:+.2f}")
    else:
        print("scorer: handcrafted eval_state")

    by_type = collections.defaultdict(list)
    by_ctrl = collections.defaultdict(list)
    alld = []
    for header, steps in games_from(args.tar, args.games):
        by_i = {s["i"]: s for s in steps}
        for s in steps:
            if not s.get("is_main"):
                continue
            me = s.get("player")
            if me is None:
                continue
            d, _s1 = build_sft._eval_delta(s, by_i, me)
            if d is None:
                continue
            s1 = _s1_step(s, by_i, me)
            passed = (s1 is not None and s1.get("player") is not None
                      and s1.get("player") != me)
            by_type[move_type(s)].append(d)
            by_ctrl["control PASSED to opp" if passed else "still MY turn"].append(d)
            alld.append(d)

    if not alld:
        print("no scorable moves found"); return
    print(f"\nscored {len(alld)} MAIN moves over {args.games} games\n")
    print(f"{'move type':12}{'n':>7}{'p10':>9}{'p50':>9}{'p90':>9}")
    for t, v in sorted(by_type.items(), key=lambda kv: -len(kv[1])):
        v = sorted(v)
        q = lambda f: v[min(len(v) - 1, int(len(v) * f))]           # noqa: E731
        print(f"{t:12}{len(v):>7}{q(.10):9.3f}{q(.50):9.3f}{q(.90):9.3f}")

    print(f"\n{'control':26}{'n':>7}{'p10':>9}{'p50':>9}{'p90':>9}")
    for t, v in sorted(by_ctrl.items()):
        v = sorted(v)
        q = lambda f: v[min(len(v)-1, int(len(v)*f))]               # noqa: E731
        print(f"{t:26}{len(v):>7}{q(.10):9.3f}{q(.50):9.3f}{q(.90):9.3f}")
    if len(by_ctrl) == 2:
        a = statistics.median(by_ctrl["control PASSED to opp"])
        b = statistics.median(by_ctrl["still MY turn"])
        print(f"\nTURN-BOUNDARY OFFSET = {a-b:+.3f} pp  (passed {a:+.3f} vs stayed {b:+.3f})")

    end = sorted(by_type.get("end turn", []))
    if end:
        off = statistics.median(end)
        print(f"\nturn-boundary offset (median delta of `end turn`) = {off:+.3f}")
        print(f"  -> suggested --eval-margin {max(0.0, -off):.2f}")
        for m in (0.0, max(0.0, -off), max(0.0, -off) * 2):
            print(f"\n  adoption at margin {m:.2f} (hard threshold):")
            for t, v in sorted(by_type.items(), key=lambda kv: -len(kv[1])):
                a = sum(1 for d in v if d >= -m)
                print(f"     {t:12} {100*a/len(v):5.1f}%  ({a}/{len(v)})")


if __name__ == "__main__":
    main()
