"""Feasibility of a card-id-first answer with a sorted tie-breaker.

Scheme: the answer is the CARD token. Where several legal options share that card and are not
the same act, a second token names which one -- but by a deterministic SORT, not by menu
position, so the second token means something the model can learn once: <s0> is always the
most-committed target.

Sort (as specified): in play before anything else, then more energy attached, then lower
remaining HP. The tie-break is computed from the PROMPT, which is exactly what the model sees, so
training and inference cannot disagree about it.

Reported: how often a second token is needed at all, where the right answer lands in the sorted
order, and how big the sub-index alphabet has to be.
"""
import gzip, json, re, sys, collections
sys.path.insert(0, ".")
from lm.action_token import equivalent

RE = re.compile(r"(?:^| )(\d+)=(\S+)")
ID = re.compile(r"(c\d+|a\d+)")
SLOT = re.compile(r"^(ACTIVE|BENCH)(\d+)$")


def first_token(o):
    """card / attack token, or the bare kind for options that name no card"""
    kind, _, rest = o.partition(":")
    if not rest:
        return "A|" + kind
    if kind == "facedown":
        return "A|facedown"
    if kind == "num":
        return "A|num|" + rest
    if kind == "attack":
        return ("a" + rest) if rest.isdigit() else rest
    m = ID.search(rest)
    return m.group(1) if m else "A|" + kind


def parse_board(prompt):
    """-> {'ACTIVE0': (energy, hp), 'BENCH0': (...), ...} for MY side"""
    m = re.search(r" ME (A\[[^\]]*\])(?: (B\[[^\]]*\]))?", prompt)
    out = {}
    if not m:
        return out
    def one(txt):
        # c741*:50/50|G3|c12 need:1 rt:1
        hp = re.search(r":(\d+)/(\d+)", txt)
        en = re.search(r"\|([A-Z](?:\d+)?(?:[A-Z]\d*)*)", txt)
        e = 0
        if en:
            for a, b in re.findall(r"([A-Z])(\d*)", en.group(1)):
                e += int(b) if b else 1
        return (e, int(hp.group(1)) if hp else 9999)
    out["ACTIVE0"] = one(m.group(1)[2:-1])
    if m.group(2):
        for i, s in enumerate(m.group(2)[2:-1].split(",")):
            out["BENCH%d" % i] = one(s)
    return out


def sort_key(o, board):
    tgt = o.split("@", 1)[1] if "@" in o else ""
    tgt = tgt.split("#")[0]
    s = SLOT.match(tgt)
    if not s:
        return (1, 0, 0, o)                       # not in play: after everything in play
    e, hp = board.get(tgt, (0, 9999))
    return (0, -e, hp, o)                         # in play, most energy, lowest HP first


n = need2 = 0
rank = collections.Counter()
width = collections.Counter()
with gzip.open("data/sft/v39_dag005.jsonl.gz", "rt") as f:
    for line in f:
        d = json.loads(line)
        t = d.get("target")
        if not t:
            continue
        opts = [o for _, o in RE.findall(d["prompt"].rsplit(":: ", 1)[-1])]
        k = int(t)
        if k >= len(opts):
            continue
        n += 1
        ft = first_token(opts[k])
        same = [o for o in opts if first_token(o) == ft]
        if len(same) < 2 or all(equivalent(o, opts[k]) for o in same):
            continue                              # one token is enough
        need2 += 1
        board = parse_board(d["prompt"])
        order = sorted(same, key=lambda o: sort_key(o, board))
        rank[order.index(opts[k])] += 1
        width[min(len(order), 8)] += 1
        if n >= 400000:
            break

print("decisions %d" % n)
print("need a second token: %d (%.2f%%)  -> expected forwards/decision %.3f"
      % (need2, 100.0 * need2 / n, 1 + need2 / n))
tot = sum(rank.values())
print("\nwhere the correct option lands in the sorted order:")
for k2 in sorted(rank):
    print("   <s%d>  %6d  %5.1f%%" % (k2, rank[k2], 100.0 * rank[k2] / tot))
print("\nhow many options share the card when a tie-break is needed:")
for k2 in sorted(width):
    print("   %s options: %6d (%.1f%%)" % ("%d" % k2 if k2 < 8 else "8+", width[k2],
                                           100.0 * width[k2] / tot))
