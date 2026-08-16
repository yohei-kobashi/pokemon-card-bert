"""Audit the v36 training data against the format the deploy path will produce.

Every prompt-format parameter is silent when wrong: no error, no size change, only a lower
win rate. So check the DATA (what the model actually learned from) rather than trusting that
the flags were passed.
"""
import gzip, json, random, re, sys
from collections import Counter

paths = sys.argv[1].split(",")
tokdir = sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 4000
MAXLEN = int(sys.argv[4]) if len(sys.argv) > 4 else 640

rows = []
for p in paths:
    with gzip.open(p, "rt") as f:
        for i, line in enumerate(f):
            if i >= N:
                break
            rows.append(json.loads(line))
print("sampled %d rows from %d file(s)" % (len(rows), len(paths)))

has_rules = sum(1 for r in rows if r["state"].startswith("RULES "))
has_deck = sum(1 for r in rows if r["state"].startswith("DECK["))
has_empty_deck = sum(1 for r in rows if r["state"].startswith("DECK[]"))
has_idme = sum(1 for r in rows if re.search(r" ID ME d_\S+", r["state"]))
has_arch = sum(1 for r in rows if re.search(r" ID ME d_\S+ a_\S+", r["state"]))
has_op = sum(1 for r in rows if " OP d_" in r["state"])
print("  RULES head          %5d  (expect 0 for glossary=none)" % has_rules)
print("  DECK[ head          %5d / %d" % (has_deck, len(rows)))
print("  DECK[] empty        %5d  (deck-out / everything seen)" % has_empty_deck)
print("  ID ME d_*           %5d / %d  (expect all)" % (has_idme, len(rows)))
print("  ID ME d_* a_*       %5d" % has_arch)
print("  OP d_*              %5d" % has_op)

# DECK[] order: canonical (ascending card id) would mean --deck-shuffle did not take effect
RE_DECK = re.compile(r"^DECK\[([^\]]*)\]")
canon = shuf = 0
orders = Counter()
for r in rows:
    m = RE_DECK.match(r["state"])
    if not m or not m.group(1):
        continue
    ids = [int(x[1:].split("x")[0]) for x in m.group(1).split(",")]
    if len(ids) < 3:
        continue
    (canon := canon + 1) if ids == sorted(ids) else (shuf := shuf + 1)
    orders[tuple(ids)] += 1
tot = canon + shuf
print("  DECK[] ascending    %5d / %d  (%.1f%%; ~0%% expected with --deck-shuffle)"
      % (canon, tot, 100.0 * canon / max(1, tot)))
print("  distinct orders     %5d / %d rows" % (len(orders), tot))

# 'remaining' mode shrinks the segment as the game goes on; 'static' does not
RE_TURN = re.compile(r"\bT(\d+)\.\d+")
by_turn = {}
for r in rows:
    m = RE_DECK.match(r["state"])
    t = RE_TURN.search(r["state"])
    if not m or not t:
        continue
    b = min(int(t.group(1)) // 5, 4)
    n = sum(int(x.split("x")[1]) if "x" in x else 1
            for x in m.group(1).split(",") if x)
    by_turn.setdefault(b, []).append(n)
print("  DECK[] card count by turn bucket (remaining mode must DECREASE):")
for b in sorted(by_turn):
    v = by_turn[b]
    print("      T%-2d-%-2d  n=%-5d  mean %.1f cards" % (b * 5, b * 5 + 4, len(v),
                                                         sum(v) / len(v)))

# pair token lengths vs the training truncation limit
from tokenizers import Tokenizer  # noqa: E402
tok = Tokenizer.from_file(tokdir + "/tokenizer.json")
tok.no_truncation()
lens = []
for r in random.Random(0).sample(rows, min(600, len(rows))):
    for c in r["candidates"][:3]:
        lens.append(len(tok.encode(r["state"], c).ids))
lens.sort()
def q(p):
    return lens[int(p * (len(lens) - 1))]
print("  pair tokens  n=%d  mean %.0f  p50 %d  p90 %d  p99 %d  max %d"
      % (len(lens), sum(lens) / len(lens), q(.5), q(.9), q(.99), lens[-1]))
over = sum(1 for x in lens if x > MAXLEN)
print("  over --max-len %d    %d / %d (%.2f%%)  <- truncated during TRAINING"
      % (MAXLEN, over, len(lens), 100.0 * over / len(lens)))
over1k = sum(1 for x in lens if x > 1024)
print("  over deploy 1024    %d / %d (%.2f%%)" % (over1k, len(lens), 100.0 * over1k / len(lens)))

lab = Counter(r.get("label") for r in rows)
print("  label field         %s" % dict(lab))
print("  cands per row       mean %.2f  max %d"
      % (sum(len(r["candidates"]) for r in rows) / len(rows),
         max(len(r["candidates"]) for r in rows)))
print("  keys                %s" % sorted(rows[0].keys()))
print()
print("SAMPLE STATE (first 400 chars):")
print(" ", rows[0]["state"][:400])
