"""A/B a CARD SWAP for one deck vs the full field (same agent pilots both lists).
    PYTHONPATH=cg-lib python tools/_swaptest.py <deck> <swaps> [games] [focus_opps]
<swaps> = comma list of id:delta, e.g. "1120:-1,1246:-1,121:+1,1182:+1"
[focus_opps] = optional comma list of opponents to report separately (bad matchups)
Validates the variant (60 cards, <=4 non-basic-energy copies, <=1 ACE SPEC).
"""
import sys, os, csv, collections
sys.path.insert(0, "tools")
import arena, library
from battle_log import load_agent
from agents._engine import _CARDS
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DECK = sys.argv[1]
SWAPS = {}
for tok in sys.argv[2].split(","):
    cid, dlt = tok.split(":")
    SWAPS[int(cid)] = SWAPS.get(int(cid), 0) + int(dlt)
GAMES = int(sys.argv[3]) if len(sys.argv) > 3 else 20
FOCUS = set(sys.argv[4].split(",")) if len(sys.argv) > 4 else set()


def _base_list():
    out = []
    with open(os.path.join(ROOT, "decks", f"{DECK}.csv")) as f:
        for r in csv.reader(f):
            if r and r[0].strip().isdigit():
                out.append(int(r[0]))
    return out


def _variant(base):
    cnt = collections.Counter(base)
    for cid, d in SWAPS.items():
        cnt[cid] += d
        if cnt[cid] < 0:
            raise ValueError(f"cannot remove {cid}: only {cnt[cid] - d} in deck")
    lst = []
    for cid, n in cnt.items():
        lst += [cid] * n
    # validate
    assert len(lst) == 60, f"variant has {len(lst)} cards"
    ace = 0
    for cid, n in cnt.items():
        c = _CARDS.get(cid)
        ct = getattr(c, "cardType", None) if c else None
        if n > 4 and ct not in (5, 6):
            raise ValueError(f"{cid} x{n} exceeds 4")
        if c and getattr(c, "aceSpec", False):
            ace += n
    if ace > 1:
        raise ValueError(f"{ace} ACE SPEC cards (max 1)")
    return lst


BASE = _base_list()
VAR = _variant(BASE)
OPPS = [d for d in library.list_decks() if d != DECK]


def _task(args):
    variant, opp = args
    deck = VAR if variant == "var" else BASE
    wa, wb = arena.match(load_agent(DECK), deck,
                         load_agent(opp), library.read_deck(opp), games=GAMES)
    return variant, opp, wa, wa + wb


def main():
    tasks = [(v, o) for v in ("base", "var") for o in OPPS]
    per = {"base": {}, "var": {}}
    tot = {"base": [0, 0], "var": [0, 0]}
    with Pool(max(1, (os.cpu_count() or 2) - 1)) as pool:
        for v, o, w, n in pool.imap_unordered(_task, tasks):
            if n:
                per[v][o] = 100 * w / n
                tot[v][0] += w; tot[v][1] += n
    b = 100 * tot["base"][0] / tot["base"][1]
    p = 100 * tot["var"][0] / tot["var"][1]
    sw = ", ".join(f"{'+' if d > 0 else ''}{d}x{cid}({_CARDS.get(cid).name if _CARDS.get(cid) else '?'})"
                   for cid, d in SWAPS.items())
    print(f"{DECK} SWAP [{sw}]")
    print(f"  overall vs field ({len(OPPS)}x{GAMES}g): base {b:.1f}%  ->  variant {p:.1f}%  "
          f"({'+' if p >= b else ''}{p - b:.1f})")
    if FOCUS:
        print("  focus matchups (base -> variant):")
        for o in sorted(FOCUS):
            if o in per["base"]:
                print(f"     {o:20} {per['base'][o]:4.0f}% -> {per['var'].get(o, 0):4.0f}%")


if __name__ == "__main__":
    main()
