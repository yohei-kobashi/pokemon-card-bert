#!/usr/bin/env python3
"""Where in the game do the errors sit, and is one deck's record count just long games?

The state string carries `T<turn>.<n>` and both prize counts (`pz`). A deck that contributes
4x the records per game is either erring 4x as often or playing games 4x as long, and those
call for opposite responses -- reweight the data, or fix the pilot.
"""
import collections
import gzip
import json
import re
import sys

T = re.compile(r"\bT(\d+)\.")
PZ = re.compile(r"\bpz(\d+)\b")


def main(path):
    rows = collections.defaultdict(lambda: {"n": 0, "t": [], "pz": [], "opz": [], "end": 0})
    for line in gzip.open(path, "rt"):
        d = json.loads(line)
        if not d["lm_was_wrong"]:
            continue
        r = rows[d["deck"]]
        r["n"] += 1
        m = T.search(d["state"])
        if m:
            r["t"].append(int(m.group(1)))
        pz = PZ.findall(d["state"])
        if len(pz) >= 2:
            r["pz"].append(int(pz[0]))
            r["opz"].append(int(pz[1]))
        if d["candidates"][d["chosen"]].startswith("end"):
            r["end"] += 1

    def med(x):
        return sorted(x)[len(x) // 2] if x else -1

    print("%-24s %7s %7s %6s %6s %6s %6s %7s"
          % ("deck", "errs", "err/gm", "medT", "maxT", "myPz", "opPz", "end%"))
    for deck, r in sorted(rows.items(), key=lambda kv: -kv[1]["n"]):
        print("%-24s %7d %7.1f %6d %6d %6.1f %6.1f %6.1f%%"
              % (deck, r["n"], r["n"] / 24.0, med(r["t"]), max(r["t"] or [0]),
                 sum(r["pz"]) / max(1, len(r["pz"])), sum(r["opz"]) / max(1, len(r["opz"])),
                 100.0 * r["end"] / r["n"]))
    tot = sum(r["n"] for r in rows.values())
    print("\ntotal %d errors over %d decks; top deck = %.1f%% of the pool"
          % (tot, len(rows), 100.0 * max(r["n"] for r in rows.values()) / tot))


if __name__ == "__main__":
    main(sys.argv[1])
