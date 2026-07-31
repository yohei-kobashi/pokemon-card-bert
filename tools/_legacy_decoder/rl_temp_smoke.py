"""Temperature smoke test for RL rollouts (docs/rl_design.md §3): pick a sampling temp
that is NEITHER too high (random play -> useless gradient) NOR too low (no exploration ->
GRPO sees no variety). Runs a SMALL rollout at each candidate temp on the same matchup
subset and reports, per temp: pilot win-rate, mean policy entropy, and the fraction of
decisions where the SAMPLED action differed from the argmax (the exploration rate).

Recommend the highest temp whose win-rate stays within `--max-drop` of the argmax(temp=0)
baseline AND whose explore-rate is in [0.15, 0.40] — enough diversity for GRPO without
degrading play. Prints a recommended --temperature for rl_loop.sh.
"""
import argparse
import math
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "cg-lib"), os.path.join(ROOT, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)


def _entropy(scores, temp):
    if temp <= 1e-6:
        return 0.0
    m = max(scores)
    ex = [math.exp((s - m) / temp) for s in scores]
    Z = sum(ex); p = [e / Z for e in ex]
    return -sum(pi * math.log(max(pi, 1e-12)) for pi in p)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="A")
    ap.add_argument("--model", required=True, help="base model id")
    ap.add_argument("--adapter", required=True, help="policy LoRA adapter dir")
    ap.add_argument("--temps", default="0.5,0.7,1.0,1.3")
    ap.add_argument("--matchups", type=int, default=12)
    ap.add_argument("--max-drop", type=float, default=0.05, help="allowed winrate drop vs temp=0")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import rl_config
    import rl_rollout as R
    cfg = rl_config.stage(args.stage)
    model = R.load_scoring(args.model, args.adapter)
    profiles = __import__("json").load(open(os.path.join(ROOT, "agents", "tuning.json")))

    pilots = cfg["pilots"]; opps = [d for d, w in cfg["opponents"].items() if w > 0]
    pairs = [(p, o) for p in pilots for o in opps if p != o]
    rng0 = random.Random(args.seed)
    pairs = rng0.sample(pairs, min(args.matchups, len(pairs)))

    temps = [0.0] + [float(t) for t in args.temps.split(",")]   # 0.0 = argmax baseline
    results = {}
    for temp in temps:
        rng = random.Random(args.seed)
        records, rewards = [], []
        for (p, o) in pairs:
            for g in range(6):                             # small: 6 games/pair
                r, n = R.play_one(p, o, model, None, temp, rng, records, profiles, g % 2 == 0)
                if r is not None:
                    rewards.append(r)
        wr = sum(1 for r in rewards if r > 0) / max(1, len(rewards))
        ent = sum(_entropy(d["scores"], temp) for d in records) / max(1, len(records))
        diff = sum(1 for d in records
                   if d["chosen"] != max(range(len(d["scores"])), key=lambda k: d["scores"][k])
                   ) / max(1, len(records))
        results[temp] = dict(winrate=wr, entropy=ent, explore=diff, n=len(rewards))
        print(f"temp={temp:>4}: winrate {wr:.1%}  entropy {ent:.2f}  "
              f"explore(sampled!=argmax) {diff:.1%}  ({len(rewards)} games)", flush=True)

    base_wr = results[0.0]["winrate"]
    ok = [t for t in temps if t > 0
          and results[t]["winrate"] >= base_wr - args.max_drop
          and 0.15 <= results[t]["explore"] <= 0.40]
    rec = max(ok) if ok else (min((t for t in temps if t > 0),
                                  key=lambda t: abs(results[t]["explore"] - 0.25)))
    print(f"RECOMMEND --temperature {rec}  "
          f"(winrate {results[rec]['winrate']:.1%} vs argmax {base_wr:.1%}, "
          f"explore {results[rec]['explore']:.1%})", flush=True)
    print(f"TEMP={rec}")            # machine-readable for rl_loop.sh


if __name__ == "__main__":
    main()
