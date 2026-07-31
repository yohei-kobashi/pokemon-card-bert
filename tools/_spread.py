"""Overall winrate + per-opponent-TYPE spread for a set of decks vs the full field.
    PYTHONPATH=cg-lib python tools/_spread.py <games> <deck1,deck2,...>
Classifies every opponent by its dominant basic-energy type (fallback: modal
energyType of its Pokemon), then reports each target deck's overall win% and its
win% broken down by opponent type, plus spread stats.
"""
import sys, os, csv, glob, statistics
sys.path.insert(0, "tools")
import arena, library
from battle_log import load_agent
from agents._engine import _CARDS
from multiprocessing import Pool

GAMES = int(sys.argv[1])
TARGETS = sys.argv[2].split(",")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TYPE = {1: "Grass", 2: "Fire", 3: "Water", 4: "Lightning",
        5: "Psychic", 6: "Fighting", 7: "Dark", 8: "Metal"}


def _deck_type(name):
    cnt = {}
    pk = {}
    with open(os.path.join(ROOT, "decks", f"{name}.csv")) as f:
        for r in csv.reader(f):
            if not r or not r[0].strip().isdigit():
                continue
            cid = int(r[0]); c = _CARDS.get(cid)
            if not c:
                continue
            if getattr(c, "cardType", None) == 5 and 1 <= getattr(c, "energyType", 0) <= 8:
                cnt[c.energyType] = cnt.get(c.energyType, 0) + 1
            if getattr(c, "cardType", None) == 0:
                et = getattr(c, "energyType", None)
                if et in TYPE:
                    pk[et] = pk.get(et, 0) + 1
    if cnt:
        return TYPE[max(cnt, key=cnt.get)]
    if pk:
        return TYPE[max(pk, key=pk.get)]
    return "Colorless"


DECKS = library.list_decks()
DTYPE = {d: _deck_type(d) for d in DECKS}


def _task(args):
    cand, opp = args
    wa, wb = arena.match(load_agent(cand), library.read_deck(cand),
                         load_agent(opp), library.read_deck(opp), games=GAMES)
    return cand, opp, wa, wa + wb


def main():
    tasks = [(c, o) for c in TARGETS for o in DECKS if o != c]
    per = {c: {} for c in TARGETS}   # per[cand][opp] = win%
    tot = {c: [0, 0] for c in TARGETS}
    with Pool(max(1, (os.cpu_count() or 2) - 1)) as pool:
        for c, o, w, n in pool.imap_unordered(_task, tasks):
            if n:
                per[c][o] = 100 * w / n
                tot[c][0] += w; tot[c][1] += n
    for c in TARGETS:
        overall = 100 * tot[c][0] / tot[c][1] if tot[c][1] else 0
        vals = list(per[c].values())
        sd = statistics.pstdev(vals) if len(vals) > 1 else 0
        print(f"\n===== {c}  ({DTYPE[c]}) =====")
        print(f"  overall vs field: {overall:.1f}%  ({tot[c][0]}/{tot[c][1]})  "
              f"| spread: sd={sd:.1f}  min={min(vals):.0f}  max={max(vals):.0f}")
        # group by opponent type
        grp = {}
        for opp, p in per[c].items():
            grp.setdefault(DTYPE[opp], []).append((p, opp))
        print("  by opponent type:")
        rows = []
        for t, lst in grp.items():
            ps = [p for p, _ in lst]
            rows.append((statistics.mean(ps), t, len(lst), min(ps), max(ps)))
        for mean, t, n, lo, hi in sorted(rows, reverse=True):
            print(f"     {mean:5.1f}%  {t:10} (n={n:2}, range {lo:.0f}-{hi:.0f})")
        # notable matchups
        best = sorted(per[c].items(), key=lambda x: -x[1])[:3]
        worst = sorted(per[c].items(), key=lambda x: x[1])[:3]
        print("  best:  " + ", ".join(f"{o} {p:.0f}%" for o, p in best))
        print("  worst: " + ", ".join(f"{o} {p:.0f}%" for o, p in worst))


if __name__ == "__main__":
    main()
