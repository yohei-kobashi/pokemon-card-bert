"""Focused higher-sample check: candidate decks vs specific targets.
    PYTHONPATH=cg-lib python tools/_vs_targets.py <games> <target1,target2,...> <cand1,cand2,...>
"""
import sys, os
sys.path.insert(0, "tools")
import arena, library
from battle_log import load_agent
from multiprocessing import Pool

GAMES = int(sys.argv[1])
TARGETS = sys.argv[2].split(",")
CANDS = sys.argv[3].split(",")


def _task(args):
    cand, tgt = args
    wa, wb = arena.match(load_agent(cand), library.read_deck(cand),
                         load_agent(tgt), library.read_deck(tgt), games=GAMES)
    return cand, tgt, wa, wa + wb


def main():
    tasks = [(c, t) for c in CANDS for t in TARGETS if c != t]
    res = {c: {} for c in CANDS}
    with Pool(max(1, (os.cpu_count() or 2) - 1)) as pool:
        for c, t, w, tot in pool.imap_unordered(_task, tasks):
            res[c][t] = (w, tot)
    hdr = "  %-20s " % "candidate" + "".join("%18s" % t for t in TARGETS) + "   avg"
    print(hdr); print("  " + "-" * (len(hdr)))
    rows = []
    for c in CANDS:
        pcts = []
        cells = ""
        for t in TARGETS:
            w, tot = res[c].get(t, (0, 0))
            p = 100 * w / tot if tot else 0.0
            pcts.append(p)
            cells += "%16s  " % ("%.1f%% (%d/%d)" % (p, w, tot))
        avg = sum(pcts) / len(pcts) if pcts else 0
        rows.append((avg, c, cells))
    for avg, c, cells in sorted(rows, reverse=True):
        print("  %-20s %s  %5.1f%%" % (c, cells, avg))


if __name__ == "__main__":
    main()
