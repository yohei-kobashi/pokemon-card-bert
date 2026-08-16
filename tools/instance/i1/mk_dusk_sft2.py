"""Carve instance1's SFT pool down to the single-deck prompt AND instance2's opponent set.

Two edits, both applied in one pass over the 40.17M-row pool:

  1. DROP the DECK[] segment. instance1 pilots one deck and one deck only, so the segment is
     a constant 57 tokens (371 -> 314, -15%) that the model has been reading as a presence
     signal rather than as contents (`deck-segment-reliance`). It is a fixed prefix between
     "[ACT]\n" and the "T<turn>." marker, which makes the edit exact rather than a guess.

  2. KEEP ONLY the opponents instance2 is training against. The pool is two layers: ~4.40M
     rows carved out of the 11-deck pilot pool, whose opponents are spread near-uniformly over
     ~55 decks (only 14.7% of them on target), and ~35.8M rows generated as dusknoir-vs-11.
     The first layer is 85% matches against decks that appear in neither the gate nor the
     ladder. Filtering costs nothing -- no re-rendering, the `opp` field is already there.

The mirror (dragapult_dusknoir) is excluded, matching instance2's own list: the protagonist
does not meet itself on the ladder.

Rows are kept on a fixed stride so the surviving mix is spread across the whole file rather
than taken from its head, which is entirely layer 1.

    python3 mk_dusk_sft2.py SRC DST [TARGET_ROWS]
"""
import gzip, json, re, sys

SRC = sys.argv[1]
DST = sys.argv[2]
TARGET = int(sys.argv[3]) if len(sys.argv) > 3 else 3_000_000

# instance2's round-6 opponent list: STAGE_C_TARGETS minus the mirror, plus slowking.
OPPS = {
    "dragapult", "marnie_grimmsnarl", "alakazam_nz", "alakazam", "crustle_geco",
    "crustle", "ogerpon_mono", "dudunsparce_box", "cynthia_garchomp", "mega_lucario_tr",
    "slowking",
}

DECK_SEG = re.compile(r"^DECK .*?(?=T\d+\.)", re.S)

import collections
kept_opp = collections.Counter()
drop_opp = collections.Counter()
n = eligible = w = miss = 0

# Pass 1 counts eligible rows so the stride can be exact; the file is read twice rather than
# guessed at once, because a stride computed from an assumed survival rate silently writes the
# wrong number of rows and there is no cheap way to notice.
with gzip.open(SRC, "rt") as f:
    for line in f:
        n += 1
        j = line.find('"opp"')
        if j < 0:
            continue
        k = line.find('"', j + 6)
        e = line.find('"', k + 1)
        opp = line[k + 1:e]
        if opp in OPPS:
            eligible += 1
            kept_opp[opp] += 1
        else:
            drop_opp[opp] += 1

stride = max(1, eligible // TARGET) if TARGET else 1
print("read %d rows | eligible %d (%.1f%%) | stride %d -> ~%d rows"
      % (n, eligible, 100.0 * eligible / max(1, n), stride, eligible // stride), flush=True)
print("KEPT opponents:")
for kk, vv in kept_opp.most_common():
    print("   %-22s %8d  %5.2f%%" % (kk, vv, 100.0 * vv / max(1, eligible)))
print("DROPPED %d rows over %d off-target opponents (top): %s"
      % (sum(drop_opp.values()), len(drop_opp),
         ", ".join("%s %d" % x for x in drop_opp.most_common(6))), flush=True)

i = 0
with gzip.open(SRC, "rt") as f, gzip.open(DST, "wt") as g:
    for line in f:
        d = json.loads(line)
        if d.get("opp") not in OPPS:
            continue
        i += 1
        if i % stride:
            continue
        st = d.get("state") or ""
        new, hit = DECK_SEG.subn("", st, count=1)
        if hit == 0:
            miss += 1
            if miss <= 3:
                print("  no DECK segment: %r" % st[:100], file=sys.stderr)
        d["state"] = new
        g.write(json.dumps(d, ensure_ascii=False) + "\n")
        w += 1

print("wrote %d rows -> %s | DECK segment missing on %d (%.3f%%)"
      % (w, DST, miss, 100.0 * miss / max(1, w)), flush=True)
if miss > w * 0.001:
    print("REFUSING: more than 0.1%% of rows had no DECK segment -- the strip is not exact",
          file=sys.stderr)
    sys.exit(1)
