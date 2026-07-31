"""Merge data-parallel rollout SHARDS into one on-policy rollout for the GRPO step.

Each shard `<s>` written by tools/rl_rollout.py is a pair:
  <s>                 gzipped decision records, one JSON object per line
  <s>.rewards.json    list of per-game {matchup, reward, n_decisions, opp_kind}

tools/rl_train._load_rollout maps records -> games by walking the rewards list and taking
`n_decisions` records per game IN ORDER. So merging = concatenate the records AND the rewards
in the SAME shard order; each shard is internally aligned, and concatenation preserves it.

Usage: python tools/merge_rollouts.py --out MERGED.jsonl.gz SHARD1.jsonl.gz SHARD2.jsonl.gz ...
"""
import argparse
import gzip
import json
import os
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="merged .jsonl.gz (also writes .rewards.json)")
    ap.add_argument("shards", nargs="+", help="shard .jsonl.gz files (each needs a .rewards.json)")
    a = ap.parse_args()

    recs_total = 0
    rewards_all = []
    used = 0
    with gzip.open(a.out, "wt") as fout:
        for sh in a.shards:
            rwp = sh + ".rewards.json"
            if not (os.path.exists(sh) and os.path.exists(rwp)):
                print(f"merge: MISSING shard {sh} (or its rewards) — SKIPPING", file=sys.stderr)
                continue
            n = 0
            with gzip.open(sh, "rt") as f:
                for line in f:
                    fout.write(line)
                    n += 1
            rw = json.load(open(rwp))
            recs_total += n
            rewards_all.extend(rw)
            used += 1
            print(f"merge: {sh}: {n} records, {len(rw)} games", file=sys.stderr)

    json.dump(rewards_all, open(a.out + ".rewards.json", "w"))
    tot = sum(g["n_decisions"] for g in rewards_all)
    status = "OK" if tot == recs_total else "MISMATCH!"
    print(f"merge: {used} shards -> {a.out}: {recs_total} records, {len(rewards_all)} games, "
          f"sum(n_decisions)={tot} [{status}]", file=sys.stderr)
    if tot != recs_total:
        raise SystemExit("merge alignment MISMATCH (records != sum n_decisions) — aborting")


if __name__ == "__main__":
    main()
