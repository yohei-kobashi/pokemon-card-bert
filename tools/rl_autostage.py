"""Decide, from a finished Stage-A run, whether to auto-continue with LARGER rounds.

Context. Stage A r0->r3 gained +2.5pt against engine_v2 and r3->r6 gained +0.35pt. Two
explanations compete: gradient noise, or the curriculum trading strong cells for weak ones
(headroom-weighted pilots + win-boost). They are distinguishable, and the data says noise:
corr(cell's delta-vs-engine at r3, its r3->r6 move) = -0.203, t=-0.90, df=19 -- not the strong
negative a trade-off would produce. Independently, two disjoint halves of one round's decisions
give update directions only 0.29 apart in cosine, i.e. a round's gradient is noise-dominated.
Bigger rounds are the matching fix; they are NOT the fix if the plateau was never real.

Prints one line: `GO <ckpt>` or `NOGO <reason>`. Exit code 0 either way (the caller branches
on the word), 2 only on a malformed run.

    python tools/rl_autostage.py <work_dir> [--stage A] [--min-gain 0.02] [--parity -0.02]
"""
import argparse
import json
import os
import re
import sys


def read_gates(work, stage):
    """[(round, delta)] in round order, from the loop's gate log."""
    path = os.path.join(work, "%s_gates.txt" % stage)
    out = []
    if os.path.exists(path):
        for line in open(path):
            m = re.search(r"\b%s r(\d+) GATE delta vs engine_v2 = (-?[\d.]+)" % stage, line)
            if m:
                out.append((int(m.group(1)), float(m.group(2))))
    out.sort()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("work")
    ap.add_argument("--stage", default="A")
    ap.add_argument("--start-delta", type=float, default=None,
                    help="the pre-RL gate; defaults to the value cached in gate_ok")
    ap.add_argument("--min-gain", type=float, default=0.02,
                    help="a last-interval gain at or above this means Stage A is STILL "
                         "climbing -- add rounds at the current size, do not pay 3x")
    ap.add_argument("--parity", type=float, default=-0.02,
                    help="delta at or above this is effectively parity: stop, do not grind")
    ap.add_argument("--print-best", action="store_true",
                    help="print only the best-gate checkpoint path and exit (the caller needs "
                         "it for every outcome, not just GO: continuing from the LAST round "
                         "would carry forward a round the gate says was worse)")
    args = ap.parse_args()

    gates = read_gates(args.work, args.stage)
    if len(gates) < 2:
        print("NOGO only %d gate(s) recorded -- Stage A did not run far enough to judge"
              % len(gates))
        return
    start = args.start_delta
    if start is None:
        p = os.path.join(args.work, "gate_ok")
        start = float(open(p).read().strip()) if os.path.exists(p) else gates[0][1]

    best_r, best_d = max(gates, key=lambda rd: rd[1])
    last_gain = gates[-1][1] - gates[-2][1]
    ckpt = os.path.join(args.work, "%s_r%d_policy" % (args.stage, best_r))

    if args.print_best:
        print(ckpt); return

    series = " ".join("r%d %+.4f" % rd for rd in gates)
    sys.stderr.write("start %+.4f | %s | best r%d %+.4f | last interval %+.4f\n"
                     % (start, series, best_r, best_d, last_gain))

    # 1. still climbing -> more rounds at the CURRENT size is cheaper than 3x rounds
    if last_gain >= args.min_gain:
        print("NOGO still improving (last interval %+.4f >= %.4f) -- add rounds, not size"
              % (last_gain, args.min_gain))
        return
    # 2. the update is harmful, not noisy -- a bigger batch just walks the wrong way faster
    if best_d < start:
        print("NOGO best gate %+.4f is below the pre-RL start %+.4f -- diagnose lr/kl/advantage"
              % (best_d, start))
        return
    # 3. essentially at parity: Stage A has done its job
    if best_d >= args.parity:
        print("NOGO best gate %+.4f is at parity (>= %.4f) -- stop, do not grind"
              % (best_d, args.parity))
        return
    # 4. the checkpoint must actually exist and be a model dir
    if not os.path.exists(os.path.join(ckpt, "config.json")):
        print("NOGO best checkpoint %s is missing or incomplete" % ckpt)
        return
    print("GO %s" % ckpt)


if __name__ == "__main__":
    main()
